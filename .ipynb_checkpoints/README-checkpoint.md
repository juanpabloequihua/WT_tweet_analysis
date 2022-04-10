
# Contents:
I. Introduuction
II. Deployment Guide 

# I. Introduction.
This project aims to deploy a streamlit application to analyse twiter data provided by WT. 


# II. Task 1: Deploy a streamlit web application to analyse tweets data

# Deployment Guide:
## Option 1 (Semi-automatic): 

1. Start terminal at folder and the file to create virtual environment, install libraries, and run the app.
(please note that this will take approximately **30 minutes** in fire up the app, as everything is done from scratch in a new virtual environment.)

```
bash Task1_streamlit_app.txt
```

## Option 2 (Manual): 

0. Start Terminal and create virtual environment:
```
pip install virtualenv
virtualenv python_wt
source python_wt/bin/activate
```

1. Start Terminal and install requirements file:
```
pip install -r requirements.txt
```

2. Run streamlit app from terminal
```
streamlit run streamlit_app.py
```

# Task 2: Convert Twiter data from csv format into json format.

Task2_Transform csv into json.ipynb python notebooko contains few lines of code to transform the data format from csv to json. The complete code to perform the format change is contained in 
```
/json_converter/converter.py
```
where the function 'get_tweet_json()' transform the complete data format. 

### Author:
Juan Pablo Equihua on April 10th, 2022.