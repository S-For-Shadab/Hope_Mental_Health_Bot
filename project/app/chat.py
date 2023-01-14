import re
import string
import numpy as np
import pandas as pd
import random
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text 

df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')

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

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

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

model = load_model("Emotion Recognition.h5")

tags = []

with open('train_dataset.json', 'r') as f:
    train_intents = json.load(f)


# loop through each sentence in our intents patterns
for intent in train_intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    
tags = sorted(set(tags))


bot_name = "HOPE"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    
    sentence = clean(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    # print(f"{result} : {proba}\n\n")

    
    for intent in train_intents['intents']:
        if result == intent["tag"]:
            print(f"{bot_name}: {random.choice(intent['responses'])}")

# sentences = [
#             "He's over the moon about being accepted to the university",
#             "Your point on this certain matter made me outrageous, how can you say so? This is insane.",
#             "I can't do it, I'm not ready to lose anything, just leave me alone",
#             "Merlin's beard harry, you can cast the Patronus charm! I'm amazed!"
#             ]
# for sentence in sentences:
#     print(sentence)
#     sentence = clean(sentence)
#     sentence = tokenizer.texts_to_sequences([sentence])
#     sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
#     result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
#     proba =  np.max(model.predict(sentence))
#     print(f"{result} : {proba}\n\n")
