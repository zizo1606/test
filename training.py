#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import  Dense, Activation , Dropout,Conv2D
from  tensorflow.keras.optimizers import Adam
from tensorflow import keras

lemmatizer = WordNetLemmatizer()

#intents = json.loads(open('intents_.json').read())
intents = json.loads(open('intents_copy.json', encoding="utf8").read())

Words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
   for pattern in intent['patterns']:
      word_list = nltk.word_tokenize(pattern)
      Words.extend(word_list)
      documents.append((word_list,intent['tag']))
      if intent['tag'] not in classes :
         classes.append(intent['tag'])

#print(documents)

Words = [lemmatizer.lemmatize(word.lower()) for word in Words if word  not in ignore_letters]

Words=sorted(set(Words))
classes=sorted(set(classes))

pickle.dump(Words , open('words.pkl','wb'))
pickle.dump(classes , open('classes.pkl','wb'))

print(len(Words))
print(len(classes))


training = []
output_empty = [0] * len(classes)

for document in documents:
   bag = []
   word_patterns = document[0]
   word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
   for word in Words:
      bag.append(1) if word in word_patterns else bag.append(0)
   
   output_row=list(output_empty)
   output_row[classes.index(document[1])] = 1
   training.append([bag, output_row])

random.shuffle((training))
training=np.array(training)

train_x = list(training[:, 0])
train_y=list(training[: , 1])

model = Sequential()
model.add(Dense(95, input_shape=(len(train_x[0]),), activation='elu',kernel_regularizer=keras.regularizers.l2(l=0.0001)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu',kernel_regularizer=keras.regularizers.l2(l=0.0001)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu',kernel_regularizer=keras.regularizers.l2(l=0.0001)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


#sgd = SGD(learning_rate=0.001, momentum=0.0, nesterov=False )
Adam = Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=Adam,metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',optimizer=Adam,metrics=['accuracy'])

history=model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=10,validation_split=0.2, verbose=1)
model.save('Chatbot.h5', history)


def plot_graphs(history, string, string2):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  plt.savefig(string2)

plot_graphs(history, "accuracy","test")
plot_graphs(history, "loss","test1")
print("Done")