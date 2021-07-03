import pandas as pd

import numpy as np

from tqdm import tqdm

from lightgbm.sklearn import LGBMClassifier

from reduce_mem import *

from auc import uAUC

import time

from gensim.models import Word2Vec

import gc

emb_size = 8 # 目前最高是8

play_cols = ['is_finish', 'play_times', 'play', 'stay']

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

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

test = pd.read_csv('wechat_algo_data1/test_b.csv') # test 在这里

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

df['stay_video'] = df['stay'] / df['videoplayseconds']  ##新3

df['play_times'] = df['play'] / df['videoplayseconds']

play_cols = [

    'is_finish', 'play_times', 'play', 'stay', 'stay_video'

]



## 统计历史天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）

n_day = 7 #可以改成7   ##新3

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

for f in tqdm(['userid', 'feedid', 'authorid', 'bgm_song_id','bgm_singer_id','tag_id']): # new2 

    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([

    ['userid', 'feedid'],

    ['userid', 'authorid'],

    ['userid', 'tag_id'],

    ['userid','bgm_song_id'],# new2 

    ['userid', 'bgm_singer_id']# new2 

]):

    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

for f1, f2 in tqdm([

    ['userid', 'authorid'],

    ['userid','bgm_song_id'],# new2 

    ['userid', 'bgm_singer_id'],# new2 

    ['userid','tag_id'] # new2 

]):

    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')

    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)

    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')

df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')

df = reduce(df)

# id embedding feature
print('start merging id embedding feature') 

tmp = pd.read_csv('data/id_embedding.csv')

tmp = reduce(tmp)

feat = [f for f in tmp.columns if f not in df.columns]

df = pd.concat([df,tmp[feat]], axis= 1)

df = reduce(df)

del tmp

gc.collect()

## 读取视频信息表
print('reading feed info')

feed_info = pd.read_csv('wechat_algo_data1/feed_info.csv')[['feedid', 'videoplayseconds']]

df = df.merge(feed_info, on='feedid', how='left')

df = reduce(df)

del feed_info

gc.collect()

# count feature
print('start making count feature')

cate_cols = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id','bgm_singer_id', 'tag_id']

for f in tqdm(cate_cols):

    tmp = df[f].map(df[f].value_counts())

    if tmp.var() > 1:

        df[f + '_count'] = tmp

# embedding feature
print('start merging embedding feature')

tmp = pd.read_csv('data/embedding_feature.csv')

df = df.merge(tmp, on= 'feedid', how= 'left')

df = reduce(df)

# pca feature
print('start merging pca feature')

feed_emb = pd.read_csv('data/feed_embeddings_PCA.csv')

feed_emb = reduce(feed_emb)

df = pd.merge(df, feed_emb, on = 'feedid', how = 'left')

df = reduce(df)

train = df[~df['read_comment'].isna()].reset_index(drop=True)

test = df[df['read_comment'].isna()].reset_index(drop=True)

cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]

print(train[cols].shape)

trn_x = train[train['date_'] < 14].reset_index(drop=True)

val_x = train[train['date_'] == 14].reset_index(drop=True)

boosting = 'dart'

##################### 线下验证 #####################
lr = 0.02  #目前0.01最好，但很慢

num_leaves = 163

uauc_list = []

reg_alpha = 2

reg_lambda = 2

r_list = []

for y in y_list[:4]:

    print('=========', y, '=========')

    t = time.time()

    clf = LGBMClassifier(

        learning_rate=lr,

        n_estimators=50000,

        num_leaves=num_leaves, #目前最好163 with 0.01

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=2021,

        metric='None',

        reg_alpha = reg_alpha,

        reg_lambda = reg_lambda
        

    )

    clf.fit(

        trn_x[cols], trn_x[y],

        eval_set=[(trn_x[cols], trn_x[y]),(val_x[cols], val_x[y])], #

        eval_metric='auc',

        early_stopping_rounds=100,

        verbose=50

    )

    val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]

    val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])

    uauc_list.append(val_uauc)

    print(val_uauc)

    r_list.append(clf.best_iteration_)

    print('runtime: {}\n'.format(time.time() - t))



weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]

print(uauc_list)

print(weighted_uauc)



##################### 全量训练 #####################

r_dict = dict(zip(y_list[:4], r_list))

for y in y_list[:4]:

    print('=========', y, '=========')

    t = time.time()

    clf = LGBMClassifier(

        learning_rate=lr,
 
        n_estimators=r_dict[y],

        num_leaves= num_leaves, #源码63

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=2021,

        reg_alpha = reg_alpha,

        reg_lambda = reg_lambda
        

    )

    clf.fit(

        train[cols], train[y],

        eval_set=[(train[cols], train[y])],

        early_stopping_rounds=r_dict[y],

        verbose=100

    )

    test[y] = clf.predict_proba(test[cols])[:, 1]

    print('runtime: {}\n'.format(time.time() - t))

test[['userid', 'feedid'] + y_list[:4]].to_csv(

    'data/submit/sub_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),

    index=False

)