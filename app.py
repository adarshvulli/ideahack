import os
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
from flask import Flask, flash, redirect, render_template, request, session, abort
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import metrics
import numpy
from model_utils import conf_keras_first_go
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import metrics
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import pandas as pd
from dataset_handler import DatasetSpliter
import requests
import json
from json import loads
def prediction(user_text):
  # load json and create model
  json_file = open('model_final.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model_final.h5")
  print("Loaded model from disk")
  
  # evaluate loaded model on test data
  loaded_model.compile(loss = 'categorical_crossentropy',
                optimizer = 'sgd',
                metrics = [metrics.categorical_accuracy, 'accuracy'])
  spliter=DatasetSpliter(conf_keras_first_go.dataset_path,conf_keras_first_go.vocab_size,conf_keras_first_go.max_length)
  split_data=spliter.data_encode()

  x_train=split_data[0]
  y_train = split_data[1]
  x_test = split_data[2]
  y_test = split_data[3]
  test_labels=spliter.test_labels
  json_file = json_file

  # Encode the text
  encoded_docs = [one_hot(user_text, conf_keras_first_go.vocab_size)]
  # pad documents to a max length
  padded_text = pad_sequences(encoded_docs, maxlen=conf_keras_first_go.max_length, padding='post')
  # Prediction based on model
  prediction = loaded_model.predict(padded_text)
  # Decode the prediction
  encoder = LabelBinarizer()
  encoder.fit(test_labels)
  final_result = encoder.inverse_transform(prediction)
  return final_result
def search(text):
  data = pd.read_csv("job_titles_IT.csv")
  title_lookup = dict(zip(data.Title, data.real))
  query = title_lookup[text]
  data_names = pd.read_csv("job_titles_names.csv")
  names_lookup = dict(zip(data_names.Title, data_names.real))
  name = names_lookup[text]
  url_get = 'https://ec.europa.eu/esco/api/resource/occupation?uris={}&language=en'.format(query)
  headers_get = {'Accept' : 'application/json'}
  resp = requests.get(url_get,headers = headers_get)
  response = resp.text
  parsedJson = loads(response)
  res = parsedJson["_embedded"][query]["_links"]["hasEssentialSkill"]
  #res = parsedJson["_embedded"][query]["_links"]["hasOptionalSkill"]
  with open('data_occupation.json', 'w') as f:
      json.dump(res, f)
  df=pd.read_json("data_occupation.json")

  df.to_csv('results_occupation.csv')
  data_csv = pd.read_csv("results_occupation.csv")
  data_new= data_csv.loc[:,["title"]]
  data_new.to_csv(r'occupation_skill.txt', header=None, index=None, sep='\t', mode='w')
  file = open("occupation_skill.txt","r")
  my_string = file.read()
  text = my_string.replace("\n", ",")
  strings = "The predicted job is {} and it's needed skills are {}".format(name,text)
  return str(strings)

 

app = Flask(__name__)


@app.route("/")
def index():

    return render_template('index.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form.getlist('Job')
      processed_text = prediction(result[0])
      result = {'Job': processed_text }
      for key, value in result.items():
        print("value",value)
        value= str(value)
        text_new = value.strip("['']")
        final = search(text_new)
        return render_template("result.html",result = final)


if __name__ == "__main__":
  app.run()
   
