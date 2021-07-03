import pandas as pd

import numpy as np

from tqdm import tqdm

from reduce_mem import *

from gensim.models import Word2Vec

emb_size = 8

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

    return tmp, tmp2

def emb(feed_df, col):

    if col == 'machine_tag_list':

        tmp = feed_df[col].apply(lambda x: str(x).replace(';', ' ').split(' '))

        tmp = tmp.apply(lambda x: [num for num in x if float(num) > 1])

    else:

        tmp = feed_df[col].apply(lambda x: str(x).split(';'))

    sentence = tmp.values.tolist()

    model = Word2Vec(sentences= sentence, min_count=5, vector_size= emb_size, window= 18)

    emb_matrix = []

    for seq in sentence:

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


feed_info = pd.read_csv('wechat_algo_data1/feed_info.csv')

df = pd.DataFrame()

df['feedid'] = feed_info['feedid']

# embedding feature
print('start making word embedding feature')

word_emb_cols = ['manual_keyword_list', 'machine_keyword_list',	'manual_tag_list', 'machine_tag_list']

for f in tqdm(word_emb_cols):

    tmp = emb(feed_info, f)

    tmp = reduce(tmp)

    df = df.merge(tmp, on = 'feedid', how = 'left')

# embedding feature
print('start making sentence embedding feature')

sentence_emb_cols = ['description', 'ocr',	'asr', 'description_char', 'ocr_char',	'asr_char'] #添加了这里

for f in tqdm(sentence_emb_cols):

    tmp = emb3(feed_info, f)

    tmp = reduce(tmp)

    df = df.merge(tmp, on = 'feedid', how = 'left')

df.to_csv('data/embedding_feature.csv', index = False)