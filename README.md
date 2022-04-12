### Author:
Juan Pablo Equihua on April 10th, 2022.

# Contents:
I. Introduuction
II. Deployment Guide 

## I. Introduction.
This project aims to deploy a streamlit application to analyse twiter data provided by WT. 


## II. Task 1: Deploy a streamlit web application to analyse tweets data

## Deployment Guide:
### Option 1 (Semi-automatic): 

1. Start terminal at folder and the file to create virtual environment, install libraries, and run the app.
(please note that this will take approximately **30 minutes** in fire up the app, as everything is done from scratch in a new virtual environment.)

```
bash Task1_streamlit_app.txt
```

### Option 2 (Manual): 

0. Start Terminal and create virtual environment:
```
pip install virtualenv
virtualenv python_wt
source python_wt/bin/activate
```

1. Start Terminal and install requirements file:
This step might take some time. 
```
pip install -r requirements.txt
```

2. Start Fine-tuned BERT API with Flask:
This command will start a BERT API to make text classification accross all 23 classes in data.clusters_0 for an user input string. API will be listening in port localhost:80 
```
python3 BERT_API/bert_api.py
```
**NOTE: If the fine-tuned BERT model is not already in the folder '/Fine_tuned_BERT/' the model can be downloaded from https://www.dropbox.com/scl/fo/18zy0zc6r8az5x7ultwoe/h?dl=0&rlkey=gqi93zce3vpcykinvk1x26egf **


2. Open a new terminal and Run streamlit app:
The app will take approximately 10 minutes to launch, as it is downloading and cashing few transformets from Hugging Face:

```
source python_wt/bin/activate
streamlit run streamlit_app.py
```

## Task 2: Convert Twiter data from csv format into json format.

```
Task2_Transform csv into json.ipynb
```
python notebook contains few lines of code to transform the data format from csv to json. The complete code to perform the format change is contained in the python file: 
```
/json_converter/converter.py
```
where the function 'get_tweet_json()' transform the complete data format. 
