#!/usr/bin/env python
# coding: utf-8


import streamlit as st
st.set_page_config(layout="wide")
from utils.utilities import *
from utils.query_similar import *
import plotly.express as px 
import tokenizers
from annotated_text import annotated_text

header = st.container()
filters = st.container()
dataset = st.container()
stats = st.container()
query_string = st.container()
bert_for_string = st.container()

ner_for_string = st.container()

data_path = './Data/data.csv'
data = load_data(data_path)
kw_extractor = yake.KeywordExtractor()

@st.cache
def preprocess_dataset(data):
    
    '''
    Function to create new sentiment for each tweet using Hugginface transformers.
    '''
    sentiment_classifier = load_sentiment_classifier()
    data = assign_transformer_sentiment(data, sentiment_classifier)
    return data

@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None}, allow_output_mutation=True)
def get_ner_model():
    return pipeline("ner", aggregation_strategy = 'simple')

ner_model = get_ner_model()

min_date = data.published_date.min()
max_date = data.published_date.max()

data = preprocess_dataset(data)

with header:
    st.title('Welcome to my DS project')
    st.subheader("""This app was developed by Juan Equihua on April 10th, 2022. The purpose of this app is analysing the Tweets dataset provided for the assigment. """)
#     st.text('The purpose of this app is analysing the Tweets dataset provided for the assigment. ')
    
    
with filters:
    st.subheader('Filter dataset by date:')
    left_col,  right_col = st.columns(2)
    min_date = pd.to_datetime(left_col.date_input('Start date', min_date))
    max_date = pd.to_datetime(right_col.date_input('End date', max_date))
    
    data = filter_data_by_date(data, min_date, max_date)
    
    topics = st.multiselect('Do you want to filter by topics and analyise more specific data? ', 
                            data.clusters_0.unique(), default = data.clusters_0.unique())
    data = filter_data_by_topic(data, topics)

    st.write('Note: Filtered data contains {} records.'.format(data.shape[0]))
    
with dataset:
    st.subheader('Analysis of the dataset:')
    st.write('This section intends to present initial visualisations of the data contained in tweeter dataset, such as the distributioon of topics, tweet sentiment obtained from the [BERT](https://github.com/huggingface/transformers) trainsformers by [Hugging Face](https://huggingface.co), and distribution of author gender.')
             
    left_col, center_col,  right_col = st.columns(3)
    
    ## Plot dataset:
    topics = pd.DataFrame(data.clusters_0.value_counts().reset_index())
    topics.columns = ['Topic', 'counts']
    topics_bar=px.bar(topics,x='counts', y='Topic', orientation='h', color_discrete_sequence=px.colors.sequential.RdBu,
                     title="Distribution of Topics")
    
    ## Gender distribution:
    gender_data = data.author_gender.value_counts().reset_index()
    gender_data.columns = ['Gender', 'counts']
    gender_plot = px.pie(gender_data, values='counts', names='Gender', 
                        color_discrete_sequence=["gray", "#00D", "pink"],
                        title="Distribution of Author Gender")
    
    ## Sentiment data:
    sentiment_data = data.transformer_sentiment.value_counts().reset_index()
    sentiment_data.columns = ['Sentiment', 'counts']
    sentiment_plot = px.pie(sentiment_data, values='counts', names='Sentiment',
                            color_discrete_sequence=px.colors.sequential.RdBu,
                           title="Distribution of Sentiments")
    
    left_col.plotly_chart(topics_bar, use_container_width=True)
    center_col.plotly_chart(sentiment_plot, use_container_width=True)
    right_col.plotly_chart(gender_plot, use_container_width=True)
    
with stats:
    stats = get_engagement_stats(data)
    st.write('Descriptive engagement stats of filtered data:')
    st.table(stats)
    
with query_string:
    st.subheader('Find the most relevants tweets in the dataset based on a string query using Gensim')
    st.write('This section intents to find the most relevant tweets in the dataset for a given query string provided by the user, query similarities are obtained with the use of the [gensim](https://pypi.org/project/gensim/) library, a software framework for topic modelling with large corpora. For each relevant tweet retrieved, we extract the top 10 keywords in the text with the use of the [Yake](https://github.com/LIAAD/yake) library.')
    
    input_sentence = st.text_input('Please type a string query', 'Economic struggle in the UK')
    
    ## Filters for number os tweets and keywords:
    left_col,  right_col = st.columns(2)
    n_tweets = int(left_col.text_input('How many similar tweetes would you like to see?', '10'))
    n_kw = int(right_col.text_input('How many keywords would you like to see', '10'))
    
    # Display similar tweets:
    similar_tweets = get_similar_tweets_to_text_string(data.body, input_sentence, rank = n_tweets, return_as_dataframe = True)
    similar_tweets['keywords'] = [keyword_extractor(kw_extractor, x, top = n_kw) for x in similar_tweets['Top similar tweets']]
    st.table(similar_tweets)
    
with bert_for_string:
    
    ## Make predictions of query classes with BERT API:
    st.subheader('Find top 3 topic predictions from Fine-tuned BERT model.')
    st.write('Predictions of Top 3 query classes (topics) are made via a a fine-tuned BERT model trained for 5 epochs with 70% of the data provided for this task. Fine-tuning code is contained in the notebook **Aux_notebook - Fine tunning BERT for multiclass.ipynb** file. The BERT model was deployed via a Flask API listening at localhost:80 and returns the top 3 predictions for the string query. ')
    
    classes = get_predictions_from_bert_api(input_sentence)
    left_col, center_col,  right_col = st.columns(3)
    top1 = classes['Top 1']
    top2 = classes['Top 2']
    top3 = classes['Top 3']
    
    left_col.write(f'- First class in your search: **{top1}**')
    center_col.write(f'- Second class in your search: **{top2}**')
    right_col.write(f'- Third class in your search: **{top3}**')

    
with ner_for_string:
    st.subheader('Find the entities names in top similar Tweets using BERT.')
    st.write("""Once we have identified the top 10 momst similar tweets in the data we are interested in identify people, locations, and organisations mentionend in the top most similar tweets, To achieve this we use the NER (Name Entity Recognition) system available from the [transformers Hugging Face](https://huggingface.co/tasks/token-classification) library with the use of [BERT](https://huggingface.co/dslim/bert-base-NER. This will identify on the fly, all entities for our top tweets and print them below:

(Please note that this process might take a while as streamlit cannot cache the NER system natively.)
    
    
    """)
    
    ## Find Entities in top similat tweets:
    ner_output_list = [NER_extractor_from_transformer(ner_model(x)) for x in similar_tweets['Top similar tweets']]
    
    list_of_people = [people[0] for people in ner_output_list]
    list_of_people  = np.unique([j for i in list_of_people for j in i])

    list_of_locations = [locations[1] for locations in ner_output_list]
    list_of_locations = np.unique([j for i in list_of_locations for j in i])

    list_of_organisations = [org[2] for org in ner_output_list]
    list_of_organisations= np.unique([j for i in list_of_organisations for j in i])
    
    left_col, center_col,  right_col = st.columns(3)
    
    left_col.write('- People mentioned in top similar queries: **{}**'.format( ', '.join(list_of_people)))
    center_col.write('- Locations mentioned in top similar queries: **{}**'.format( ', '.join(list_of_locations)))
    right_col.write('- Organisations mentioned in top queries: **{}**'.format( ', '.join(list_of_organisations) ))
    

    
    