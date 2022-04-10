#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
# import tensorflow as tf
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
import yake
import streamlit as st
from transformers import pipeline


def load_sentiment_classifier():
    return pipeline('sentiment-analysis')

def load_data(path):
    
    ''' Function to load data into a Pandas DataFrame and clean date '''
    data = pd.read_csv(path)
    data.published_date = pd.to_datetime(data.published_date)
    
    return data

def filter_data_by_topic(data, topics):
    ''' filter data by topics contained in column cluster_0'''
    return data[data.clusters_0.isin(list(topics))].reset_index(drop = True)

def filter_data_by_date(data, min_date, max_date):
    ''' Funtion to filter the dataset by a given date '''
    data = data[(data.published_date>=min_date)&(data.published_date<=max_date)]
    return data

def get_engagement_stats(data):
    
    ''' Function to extract stats of engagement data'''
    
    temp_data = data[['total_engagement', 'comments', 'shares', 'likes']].describe().loc[['mean','max']]
    return temp_data

def keyword_extractor(kw_extractor, text, top = 10):
    
    '''
    Extract the top n key words of a text with Yake library.
    https://github.com/LIAAD/yake
    '''
    keywords = kw_extractor.extract_keywords(text)
    keywords_list = [x[0] for x in keywords[:top]]
    
    return keywords_list


def transformer_sentiment(sentiment_classifier, text):
    '''Predict sentiment of any text.  '''
    return sentiment_classifier(text)[0]['label']


def assign_transformer_sentiment(data, sentiment_classifier):
    ''' Assign new sentiment label to each tweet in the dataset. '''
    ## Use pre-trained transformer 'pipeline' from Huggingface to assign new sentiment to each tweet.
    ## Ref: https://github.com/huggingface/transformers
    
    data['transformer_sentiment'] = [transformer_sentiment(sentiment_classifier, x) for x in data.body]
    return data

def NER_extractor_from_transformer(ner_dictionary):
    ''' Clean ner_dictionary_output and export NER results as tuples'''
    people = []
    locations = []
    organizations = []
    
    if len(ner_dictionary) ==0:
        return (people, locations, organizations)
    
    else: 
        try:
            for token in range(len(ner_dictionary)):
                if ner_dictionary[token]['entity_group'] == 'PER': 
                    people.append(ner_dictionary[token]['word'])

                elif ner_dictionary[token]['entity_group'] == 'LOC':
                    locations.append(ner_dictionary[token]['word'])

                elif ner_dictionary[token]['entity_group'] == 'ORG':
                    organizations.append(ner_dictionary[token]['word'])
                else:
                    pass
        except: 
            return (people, locations, organizations)
                
    return (people, locations, organizations)

