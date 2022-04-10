#!/usr/bin/env python
# coding: utf-8
import pandas as pd


def load_data(path):

    """Function to load data into a Pandas DataFrame and clean date"""

    data = pd.read_csv(path)
    data.published_date = pd.to_datetime(data.published_date, utc=False)
    data = data.reset_index()
    data.rename(columns={"index": "Tweet_id"}, inplace=True)

    return data


def fix_date_format(date_string):
    """Fix the date format from string into dd/mm/yyyy."""

    year = date_string[:4]
    month = date_string[5:7]
    day = date_string[-2:]

    return f"{day}/{month}/{year}"


def get_json_metrics(row):
    """Obtain engagement metrics from a single row."""

    return {
        "total_engagement": int(row.total_engagement),
        "comments": int(row.comments),
        "shares": int(row.shares),
        "likes": int(row.likes),
    }


def get_tweet_info(row):
    """Obtain Tweeet information for a singe row."""

    return {
        "tweet_id": int(row.Tweet_id),
        "type": row.post_type,
        "tweet": row.body,
        "sentiment": row.sentiment_summary,
        "metrics": [get_json_metrics(row)],
        "url": "...",
    }


def get_tweets_for_date(data, date):
    """Obtain information of tweets published in a single date by calling the function get_tweet_info in a loop."""

    for i in range(2):
        temp_data_filtered_by_date = data[data.published_date == date]
        tweets = []

        for j in range(temp_data_filtered_by_date.shape[0]):
            tweets.append(get_tweet_info(temp_data_filtered_by_date.iloc[j]))

    return {"tweet_created:": fix_date_format(str(date)[:10]), "tweets": tweets}


def get_tweet_json(data):
    """Obtain the final tweet JSON format for a dataset."""

    unique_dates = data.published_date.unique()
    tweets = []

    for date in unique_dates:
        tweets.append(get_tweets_for_date(data, date))

    return {"Huawei": {"Twitter": tweets}}
