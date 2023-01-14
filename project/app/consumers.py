from channels.exceptions import StopConsumer
from channels.consumer import SyncConsumer
from time import sleep
import asyncio
import cv2
from deepface import DeepFace
from django.shortcuts import render
from http.client import HTTPResponse
from django.shortcuts import render
from collections import Counter

import re
import string
import numpy as np
import pandas as pd
import random
from random import choice
import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# from tf.keras.utils import to_categorical


str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text 

df_train = pd.read_csv('app\\train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('app\\val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('app\\test.txt', names=['Text', 'Emotion'], sep=';')

X_train = df_train['Text'].apply(clean)
y_train = df_train['Emotion']

X_test = df_test['Text'].apply(clean)
y_test = df_test['Emotion']

X_val = df_val['Text'].apply(clean)
y_val = df_val['Emotion']

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

# Tokenize words
tokenizer = Tokenizer()
# Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))


sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(sequences_train, maxlen=256, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=256, truncating='pre')
X_val = pad_sequences(sequences_val, maxlen=256, truncating='pre')

model = load_model("app\\Emotion Recognition.h5")

tags = []

with open('app\\train_dataset.json', 'r') as f:
    train_intents = json.load(f)


# loop through each sentence in our intents patterns
for intent in train_intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    
tags = sorted(set(tags))


#generating reponse from model
def generate(x):
        
        name=x
        sentence=name
        sentence = clean(sentence)
        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
        result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
        proba =  np.max(model.predict(sentence))
        name=result
        # botname="Hope"
        print("Statement emotion detected: ", result)
        for intent in train_intents['intents']:
            if result == intent["tag"]:
                return  random.choice(intent['responses'])



#emotion recognizing function
def facecam():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(1) #to check whether webcam is opened correctly
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    l1=[]


    while True:
        ret,frame = cap.read() # will read one image from a video
        enforce_detection = False
        result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection = False)
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #print(faceCascade.empty())
        faces = faceCascade.detectMultiScale(gray,1.1,4)
        
        #Draw a rectangular frameq
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #Use putText() method for
        #inserting text on video
        
        l1.append(result['dominant_emotion'])

        cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        
        cv2.imshow('Original video', frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return l1
emotion_start=""
emotion_end=""
class MySyncConsumer(SyncConsumer):
    def websocket_connect(self,event):
        
        print('Websocket connected...',event)
        self.send({
            'type':'websocket.accept',
        })
        global emotion_start
        list_start=facecam()
        a=dict(Counter(list_start))
        b = [i for i in a if a[i]==max(a.values())]
        emotion_start+=b[0]
        



    def websocket_receive(self,event):
        
        print("User: ", event['text'])

        if event['text']!="quit":

            
            response=generate(event['text'])
            print("HOPE BOT: ", response)
            print('\n')
        
        else:
            response="Bye see you later"
            print("HOPE BOT: ", response)
            print('\n')
            print(">>>>emotion at start",emotion_start)
            
            
            global emotion_end
            list_end=facecam()
            a=dict(Counter(list_end))
            b = [i for i in a if a[i]==max(a.values())]
            emotion_end+=b[0]

            print(">>>>emotion at end",emotion_end)
           
            


        self.send({
                    'type':'websocket.send',
                    'text':str(response)
            })   
            
            
        

    
    def websocket_disconnect(self,event):
        print('Websocket Disconnected...',event)
        raise StopConsumer()

    


