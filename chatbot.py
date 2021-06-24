# -*- coding: utf-8 -*-

import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()


#intents = json.loads(open('intents.json').read())
intents = json.loads(open( 'intents_copy.json' , encoding="utf8").read())

Words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl' , 'rb'))

model = load_model('Chatbot.h5')
#model1 = load_model('c:/Users/zi_errabi/Desktop/Chatbot/Chatbot/bot/Chatbot1.h5')
#model2 = load_model('c:/Users/zi_errabi/Desktop/Chatbot/Chatbot/bot/Chatbot2.h5')


def clean_up_sentence(sentence):
    sentence_words= nltk.word_tokenize(sentence)
    sentence_words= [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag= [0]*len(Words)
    for w in sentence_words:
        for i, word in enumerate(Words):
            #print(i,word)
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    #print(bow)
    res = model.predict(np.array([bow]))[0]
    #print(res)
    #print(model.predict(np.array([bow])))
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r> ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent' : classes[r[0]],'probability' : r[1]})
    print(return_list)
    return return_list



def get_response(categ_list , categ_json):

    if categ_list == []:
         a="Désolé, j'arrive pas a vous repondre , merci de bien m'expliquer ou contacter le numero 066650950, Merci"  
    else:
        for i in range(len(categ_list)):
            #print(len(categ_list))
            if categ_list[i]['probability'] > 0.50:
                tag= categ_list[i]['intent']
                #print(tag)
                list_of_intents = categ_json['intents']
                for j in list_of_intents:
                    if j['tag'] == tag:
                        a = random.choice(j['response'])
                        return a
    a="Désolé, j'arrive pas a vous repondre , merci de bien m'expliquer ou contacter le numero 066650950, Merci"  
    
    return a


