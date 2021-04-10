# Title     : TODO
# Objective : TODO
# Created by: Little Jennie Fairy
# Created on: 2021/4/2
#%%
import pickle as pkl
import bitermplus as btm
import numpy as np
from gzip import open as gzip_open
import pandas as pd

data = pd.read_csv('../data/depression_data.csv',encoding='utf-8-sig', dtype={"thread_id": str, 'replied': str})
user = pd.read_csv('../data/user_sw400.csv',encoding='utf-8-sig')
thread = pd.read_csv('../data/thread_sw2000.csv',encoding='utf-8-sig')

texts = data['space_sw']
user_texts = user['space_sw']
thread_texts = thread['space_sw']

top_n = 30
topic_range = range(2,4)
perplexity_list = []

# Vectorizing documents, obtaining full vocabulary and biterms
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
biterms = btm.get_biterms(docs_vec)

for topic_number in topic_range:
    # Initializing and running model
    model = btm.BTM(
        X, vocabulary, seed=12321, T=topic_number, W=vocabulary.size, M=20, alpha=50/topic_number, beta=0.01)
    model.fit(biterms, iterations=20)
    p_zd = model.transform(docs_vec, 'mix')

    # Calculating metrics
    perplexity = model.perplexity_
    perplexity_list.append(perplexity)
    # print(perplexity)
    # coherence = model.coherence_
    # print(coherence)

    words = []
    sort_index = np.argsort(-model.matrix_topics_words_, axis=1)
    for z in sort_index:
        sort_words = []
        for i in z[:top_n]:
            w = model.vocabulary_[i]
            sort_words.append(w)
        words.append(sort_words)
    print(words)

    with open('model{}.pickle'.format(topic_number), 'wb') as file:
        pkl.dump(model, file)

    # topic = pd.DataFrame()
    # topic['portrait'] = data['portrait']
    # topic['topic'] = list(p_zd)
    # topic.to_csv('all_biterm_topic{}.csv'.format(topic_number), encoding='utf-8-sig', index=False)


perplexity_df = pd.DataFrame()
perplexity_df['topic_num'] = topic_range
perplexity_df['perplexity'] = perplexity_list
perplexity_df.to_csv('perplexity_biterm.csv',encoding='utf-8-sig')

best_topic = perplexity_df['topic_num'][perplexity_df['perplexity'].idxmin()]
print('best_topic_number:',best_topic)

with open('model{}.pickle'.format(best_topic), 'rb') as f:
    best_model = pkl.load(f)

user_vec = btm.get_vectorized_docs(user_texts, vocabulary)
user_td = best_model.transform(user_vec, 'mix')
user_topic = pd.DataFrame()
user_topic['portrait'] = user['portrait']
user_topic['topic'] = list(user_td)
user_topic.to_csv('user_biterm_topic{}.csv'.format(best_topic), encoding='utf-8-sig', index=False)

thread_vec = btm.get_vectorized_docs(thread_texts, vocabulary)
thread_td = best_model.transform(thread_vec, 'mix')
thread_topic = pd.DataFrame()
thread_topic['thread_id'] = thread['thread_id']
thread_topic['topic'] = list(thread_td)
thread_topic.to_csv('thread_biterm_topic{}.csv'.format(best_topic), encoding='utf-8-sig', index=False)
