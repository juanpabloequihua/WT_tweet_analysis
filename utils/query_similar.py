#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from gensim import corpora, models, similarities
import pandas as pd

def get_similar_tweets_to_text_string(documents, text_string, rank = 10, return_as_dataframe = False):

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    
    vec_bow = dictionary.doc2bow(text_string.lower().split())
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    
    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
    index.save('/tmp/deerwester.index')
    index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    
    sims = index[vec_lsi] 
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    
    if return_as_dataframe:
        top_similar = [x[0] for x in sims[:rank]]
        top_df = pd.DataFrame(documents[top_similar]).reset_index(drop = True)
        top_df.columns=['Top similar tweets']
        return top_df

    else:
        for doc_position, doc_score in sims[:rank]:
            print('Similarity: ', np.round(doc_score, 3), 'Text: ', documents[doc_position])
            
            