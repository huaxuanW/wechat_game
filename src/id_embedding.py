import pandas as pd

import numpy as np

from tqdm import tqdm 

from reduce_mem import *

from gensim.models import Word2Vec

emb_size = 8

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

def emb2(df, f1, f2):
    
    df = df.fillna(0)

    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})

    sentences2 = tmp.pop('{}_{}_list'.format(f1, f2))

    sentences2 = sentences2.values.tolist()

    for i in range(len(sentences2)):
        
        sentences2[i] = [str(x) for x in sentences2[i]]
    
    model = Word2Vec(sentences= sentences2, min_count=1, vector_size= emb_size, window= 6)

    emb_matrix = []

    for seq in sentences2:

        vec = []

        for w in seq:

            if w in model.wv.key_to_index:

                vec.append(model.wv[w])

        if len(vec) > 0:

            emb_matrix.append(np.mean(vec, axis=0))

        else:

            emb_matrix.append([0] * emb_size)

    emb_matrix = np.array(emb_matrix)

    for i in range(emb_size):
        
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]

    tmp[f'{f1}_{f2}_emb_average'] = np.average(emb_matrix, axis=1)  # new 1

    tmp[f'{f1}_{f2}_emb_prod'] = np.prod(emb_matrix, axis=1) # new 2 可能需要删掉这个

    word_list = []

    emb_matrix2 = []

    for w in model.wv.key_to_index:

        word_list.append(w)

        emb_matrix2.append(model.wv[w])

    emb_matrix2 = np.array(emb_matrix2)

    tmp2 = pd.DataFrame()

    tmp2[f2] = np.array(word_list).astype(float).astype(int)

    for i in range(emb_size):

        tmp2['{}_{}_emb_{}'.format(f2, f1, i)] = emb_matrix2[:, i]

    tmp2[f'{f1}_{f2}_emb_average'] = np.average(emb_matrix2, axis=1) # new 1

    tmp2[f'{f1}_{f2}_emb_prod'] = np.prod(emb_matrix2, axis=1) # new 2 可能需要删掉这个

    return tmp, tmp2





# id embedding feature
print('start making id embedding feature')    

train = pd.read_csv('wechat_algo_data1/user_action.csv')

test = pd.read_csv('wechat_algo_data1/test_b.csv') # test 在这里

df = pd.concat([train, test], axis=0, ignore_index=True)

feed_info = pd.read_csv('wechat_algo_data1/feed_info.csv')

df = df.merge(feed_info[['feedid', 'authorid','bgm_song_id','bgm_singer_id']], on = 'feedid', how = 'left')

# popular tag
print('start making popular tag feature')

col = 'machine_tag_list'

tmp = pd.DataFrame()

tmp['feedid'] = feed_info['feedid']

tmp['tag_id'] = feed_info[col].apply(lambda x: tag_col(x))

tmp = reduce(tmp)

df = df.merge(tmp, on = 'feedid', how = 'left')

cols = ['userid', 'feedid', 'authorid', 'tag_id', 'bgm_song_id','bgm_singer_id']

df2 = pd.DataFrame()

df2[cols] = df[cols]

id_emb_cols = [['userid', 'feedid'], ['userid', 'authorid'], ['userid', 'tag_id'],['userid', 'bgm_song_id'], ['userid','bgm_singer_id']] # new 1

for f1, f2 in tqdm(id_emb_cols):
    
    tmp , tmp2 = emb2(df, f1, f2)

    df2 = df2.merge(tmp, on=f1, how='left').merge(tmp2, on=f2, how='left')
    
df2 = reduce(df2)

print(df2.shape)

df2.to_csv('data/id_embedding.csv', index = False)