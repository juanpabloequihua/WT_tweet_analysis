
# Import Flask and Machine Learning Dependencies for 
# loading your model and preparing the input data
from flask import Flask, request
import numpy as np
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification, DefaultDataCollator
import tensorflow as tf
# ...

# Create a function to load your model 
def load_model(model_path): 
    
    ''' Load fine-tined BERt model'''
    return TFAutoModelForSequenceClassification.from_pretrained(model_path)

def load_tokenizer():
    ''' Load tokenizer '''
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def make_predictions_for_string(string, tokenizer, model, labels):
    ''' Make Top 3 class predictions from string query '''
    
    tokenized_string_test = tokenizer.encode(string,
                                            truncation=True,
                                            padding=True,
                                            return_tensors="tf")
    prediction = model(tokenized_string_test)[0]
    prediction_probs = tf.nn.softmax(prediction,axis=1).numpy()
    
    sorted_prob_index = np.argsort(-prediction_probs)[0]
    
    return {"Top 1": labels[sorted_prob_index[0]], "Top 2": labels[sorted_prob_index[1]], "Top 3": labels[sorted_prob_index[2]]}

model_path = './Fine_tuned_BERT/multiclassification_bert'
# Load the model
classifier = load_model(model_path)
tokenizer = load_tokenizer()


### SETTING UP FLASK APP AND FLAKS ENDPOINTS ###
# Create the flaks App
app = Flask(__name__)

# Define an endpoint for calling the predict function based on your ml library/framework
@app.route("/bert", methods=["GET","POST"])
def predict():
    # Load the Input
    string = request.form.to_dict()['text']
    print(string)
    
    labels = ['Poor Pay', 'Cost of Living', 'Wage Growth', 'Rich People', 'Low Income Families', 'Public Sector Pay', 'Government Support', 'Mental Health', 'Leaseholding',
              'State Pension', 'Pay Rises', 'Long Hours', 'Income Tax', 'Poor People', 'Council Tax', 'Small Businesses', 'Statutory Sick Pay', 'Social Care', 
              'House Prices', 'Job', 'Minimum Wage Increase', 'National Insurance', 'Gender Pay Gap']
    
    # Make predictions on input data
    predictions_dict = make_predictions_for_string(string, tokenizer, classifier, labels)
    
    return predictions_dict
    
# Start the flask app and allow remote connections
app.run(host='0.0.0.0', port = 80)