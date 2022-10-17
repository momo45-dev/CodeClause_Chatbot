
## Import essential libaries ##

import random
import json
import numpy as np
import pickle
import nltk
import tensorflow
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer= WordNetLemmatizer()
intents = json.loads(open('D:\Desktop\C++\Chatbot\intents.json').read())

words= pickle.load(open('D:\Desktop\C++\Chatbot\words.pkl', 'rb'))
classes= pickle.load(open('D:\Desktop\C++\Chatbot\classes.pkl', 'rb'))
model=load_model('chatbotmodel.h5')

## Now we define 4 functions that are essential for our chatbot ##

## this function is reasponsible for text preprocessing ##

def sentence_cleaner(sentence):
    sentence_words= nltk.word_tokenize(sentence)
    sentence_words= [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

## in this function we perform vectorisation ##
def bag_of_words(sentence):
  sentence_words= sentence_cleaner(sentence)
  bag= [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word==w:
        bag[i]=1
  return np.array(bag)

## here we predict the class of the sentence to later be able to determine the best response ##
def predict_class(sentence):
  bow= bag_of_words(sentence)
  res= model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD=0.25
  results= [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
  results.sort(key=lambda x:x[1], reverse=True)
  return_list=[]
  for r in results :
    return_list.append({'intent': classes[r[0]],'probability': str(r[1])})
  return return_list

## after we clean, vectorise and predict the class of the sentence , now we predict a suitable response ##

def get_response(intents_list, intents_json):
  tag= intents_list[0]['intent']
  list_of_intents= intents_json['intents']
  for i in list_of_intents:
    if i['tag']== tag:
      result=random.choice(i['responses'])
      break
  return result

print("go go go")


## As long as the user is giving input
## the chatbot will keep working

while True:
  message=input("")
  ints=predict_class(message)
  res= get_response(ints,intents)
  print(res)