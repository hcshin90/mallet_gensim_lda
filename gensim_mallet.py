#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2018. 03
@author

hyunchulshin
alsdok@naver.com
"""

#from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
import gensim
from gensim import corpora
import numpy as np
import pickle as pk
import logging
import time
import ldamallet_params

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[]

f = open(ldamallet_params.input_file_path, 'r', encoding = 'utf8')

txt = f.read()
txtn = txt.split('\n')
txtn = txtn[:-1]

texts = []

dt_load_start = time.time()
print('Data Loading..')
for i, nn in enumerate(txtn):
        
    nnt = nn.split('\t')
    texts.append(nnt[-1].split(' '))
dt_load_finish = time.time()

print('Data Loading Time Usage : ' + '%.2f' % (dt_load_finish - dt_load_start) + ' seconds')

# In[]
    
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Corpora_and_Vector_Spaces.ipynb
# 위 주소는 gensim 패키지에서 사용하는 인풋 형태를 만들어가는 과정의 Tutorial임

# dictionary는 texts에 만들어 놓은 각각의 토큰들의 중복을 제거하여
# 사전 형태로 인덱스를 붙여놓은 것임

'''
[['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
'''

# 텍스트의 형태가 위와 같았다면,
# 딕셔너리는 아래와 같이 유니크한 것들만 모아서 인덱스를 붙인 것
'''
Dictionary(12 unique tokens: ['human', 'interface', 'computer', 'survey', 'user']...)
'''

print('Making Dictionary, Corpus ..')

dictionary = corpora.Dictionary(texts)
dictionary.save(ldamallet_params.output_file_path + 'dictionary_.dict')  # store the dictionary, for future reference

# 딕셔너리 확인
#print(dictionary)
#print(dictionary.token2id)

# 딕셔너리에 새로운 단어를 추가하는 방법은 생략하였음(튜토리얼 참고)
corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use

# In[]

model_start = time.time()

print('Making LDA Model using LdaMallet..')
model = LdaMallet(ldamallet_params.mallet_path, corpus = corpus, num_topics = ldamallet_params.topic_num, id2word=dictionary, workers = ldamallet_params.worker_num, iterations = ldamallet_params.iterations, prefix = ldamallet_params.output_file_path)

'''
    mallet_path (str) – Path to the mallet binary, e.g. /home/username/mallet-2.0.7/bin/mallet.
    corpus (iterable of iterable of (int, int), optional) – Collection of texts in BoW format.
    num_topics (int, optional) – Number of topics.
    alpha (int, optional) – Alpha parameter of LDA.
    id2word (Dictionary, optional) – Mapping between tokens ids and words from corpus, if not specified - will be inferred from corpus.
    workers (int, optional) – Number of threads that will be used for training.
    prefix (str, optional) – Prefix for produced temporary files.
    optimize_interval (int, optional) – Optimize hyperparameters every optimize_interval iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization).
    iterations (int, optional) – Number of training iterations.
    topic_threshold (float, optional) – Threshold of the probability above which we consider a topic.
'''

model_finish = time.time()
print('Data Loading Time Usage : ' + '%.2f' % ((model_finish - model_start)/60) + ' mins')

# In[]

print('\nSaving Model, Dictionary, Corpus..')

# 모델 저장
# gensim.modelswrappers.ldamallet.malletmodel2ldamodel 메소드는
# 현재 버전에서 버그가 발생함. (mallet model과 gensim ldamodel을 동일하게 매칭시켜주지 못함)
# gensim_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)

# Perhaps this is behavior by design, since the topic assignment of new documents seems to work.
# I don't see where the distribution is copied to the new model, though, and passing the distribution via the "eta" parameter
# in the call to LdaModel gives a (somewhat) better result (see snippet below)

def malletmodel2ldamodel(mallet_model, gamma_threshold=0.001, iterations=50):
    """
    Function to convert mallet model to gensim LdaModel. This works by copying the
    training model weights (alpha, beta...) from a trained mallet model into the
    gensim model.
    Args:
    mallet_model : Trained mallet model
    gamma_threshold : To be used for inference in the new LdaModel.
    iterations : number of iterations to be used for inference in the new LdaModel.
    Returns:
    model_gensim : LdaModel instance; copied gensim LdaModel
    """
    model_gensim = gensim.models.ldamodel.LdaModel(
      id2word = mallet_model.id2word, num_topics=mallet_model.num_topics,
      alpha = mallet_model.alpha, iterations=iterations,
      eta = mallet_model.word_topics,
      gamma_threshold = gamma_threshold,
      dtype = np.float64 # don't loose precision when converting from MALLET
    )
    model_gensim.expElogbeta[:] = mallet_model.wordtopics
    return model_gensim

gensim_ = malletmodel2ldamodel(model)
gensim_.save(ldamallet_params.output_file_path + 'gensim_model')

# corpus, dictionary 저장 (pyvis 사용을 위함)
with open(ldamallet_params.output_file_path + 'corpus_.pickle', 'wb') as f:
    pk.dump(corpus, f)

with open(ldamallet_params.output_file_path + 'dictionary_.pickle', 'wb') as f:
    pk.dump(dictionary, f)

# In[]
    
## model을 생성하면 아웃풋 파일들이 prefix 파라미터에서 지정한 './' 경로에 저장됨
## 그 중에 doctopics.txt는 각 문서들이 해당 토픽일 확률을 계산한 파일임
## 아래 코드는 doctopics.txt에서 각 문서들에게 가장 높은 확률을 갖는 토픽을
## 할당하여 추가한 파일(doctopics_assigned.txt)를 만들어 저장하는 작업임

print('Assigning Topic to Each Documents..')
f = open(ldamallet_params.output_file_path + 'doctopics.txt')
ff = open(ldamallet_params.output_file_path + 'doctopics_assigned.txt', 'w', encoding = 'utf8')

doct = f.read().split('\n')
f.close()

for i, doc in enumerate(doct):
    
    if doc == '':
        print('idx', i+1, '/', len(doct), 'is empty.  loop is stopped.. (Usually last line is empty.)')
        break
    doc_temp = doc.split('\t')
    ff.write(doc_temp[0] + '\t' + str(doc_temp[2:].index(max(doc_temp[2:]))))
    
    for j in range(model.num_topics):
        ff.write('\t' + doc_temp[-model.num_topics + j])
    
    ff.write('\n')
    
ff.close()

print('finished')
