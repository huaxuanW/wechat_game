import pandas as pd

import numpy as np

from tqdm import tqdm

from reduce_mem import *

from gensim.models import Word2Vec

emb_size = 8 # 目前最高是8

def emb3(feed_df, col):

    tmp = feed_df[col].apply(lambda x: str(x).split(' '))

    sentences3 = tmp.values.tolist()

    model = Word2Vec(sentences= sentences3, min_count=5, vector_size= emb_size, window= 18)

    emb_matrix = []

    for seq in sentences3:

        vec = []

        for w in seq:

            if w in model.wv.key_to_index:

                vec.append(model.wv[w])

        if len(vec) > 0:

            emb_matrix.append(np.mean(vec, axis=0))

        else:

            emb_matrix.append([0] * emb_size)
    
    emb_matrix = np.array(emb_matrix)

    tmp = pd.DataFrame()

    tmp['feedid'] = feed_df['feedid']
    
    for i in range(emb_size):
	
    	tmp[f'{col}_emb_{i}'] = emb_matrix[:, i]
    
    tmp[f'{col}_emb_average'] = np.average(emb_matrix, axis=1) # new 1

    return tmp

def tag_col(x):
    if type(x) != str:
        return 0
    else:
        x = str(x).replace(';',' ').split(' ')
        tag = []
        prob = []

        for i in x:
            if float(i) > 1:
                tag.append(i)
            else:
                prob.append(float(i))
        idx = prob.index(max(prob))
        pop_tag = tag[idx]
        return int(pop_tag)


y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

max_day = 15



## 读取训练集

train = pd.read_csv('wechat_algo_data1/user_action.csv')

train = reduce(train)

print(train.shape)

## 读取测试集

test = pd.read_csv('wechat_algo_data1/test_b.csv')

test['date_'] = max_day

test = reduce(test)

print(test.shape)

## 合并处理

df = pd.concat([train, test], axis=0, ignore_index=True)


## 读取视频信息表

feed_info = pd.read_csv('wechat_algo_data1/feed_info.csv')

df = df.merge(feed_info[['feedid', 'authorid', 'videoplayseconds','bgm_song_id','bgm_singer_id']], on='feedid', how='left')

df = reduce(df)

# popular tag
print('start making popular tag feature')

col = 'machine_tag_list'

tmp = pd.DataFrame()

tmp['feedid'] = feed_info['feedid']

tmp['tag_id'] = feed_info[col].apply(lambda x: tag_col(x))

df = df.merge(tmp, on = 'feedid', how = 'left')

df = reduce(df)

## 视频时长是秒，转换成毫秒，才能与play、stay做运算

df['videoplayseconds'] *= 1000

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）

df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')

df['play_times'] = df['play'] / df['videoplayseconds']

play_cols = [

    'is_finish', 'play_times', 'play', 'stay'

]



## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）

n_day = 5 #可以改成7

for stat_cols in tqdm([

    ['userid'],

    ['feedid'],

    ['authorid'],

    ['device'], # new

    ['bgm_singer_id'], # new

    ['bgm_song_id'], # new

    ['tag_id'], # new

    ['userid', 'authorid'], # new

    ['userid', 'bgm_singer_id'], # new

    ['userid', 'bgm_song_id'], # new

    ['userid', 'tag_id'] # new

]): # 可以加入device， bgm singer， bgm song，和tag

    f = '_'.join(stat_cols)

    stat_df = pd.DataFrame()

    for target_day in range(2, max_day + 1):

        left, right = max(target_day - n_day, 1), target_day - 1

        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)

        tmp['date_'] = target_day

        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)

        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')

        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:

            for stat in ['max', 'mean']: # min, median std

                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)

                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in y_list[:4]: # count

            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')

            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')

            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])



        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)

        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        print(f'working on day {target_day}')

        stat_df = reduce(stat_df)

        del g, tmp

    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')

    del stat_df

    gc.collect()


## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

for f in tqdm(['userid', 'feedid', 'authorid']):

    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([

    ['userid', 'feedid'],

    ['userid', 'authorid'],

    ['userid', 'tag_id']

]):

    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

for f1, f2 in tqdm([

    ['userid', 'authorid'] # userid-bgm_song, userid-bgm_singer, userid-tag

]):

    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')

    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)

    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')

df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')

df = reduce(df)

df.to_csv('data/stat_feature.csv', index=False)