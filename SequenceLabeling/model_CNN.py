
# coding: utf-8

#from scikits.statsmodels.tools import categorical
#b = categorical(a, drop=True)
#keras.utils.to_categorical(y, num_classes=39)

import pandas as pd
import numpy as np
import sys

raw = pd.read_csv('mfcc/train.ark',sep=' ',header=None)

label = pd.read_csv('train.lab',header=None)


data = pd.merge(raw, label,how='left',on=[0])#.iloc[:,-1]

map_39 = pd.read_csv('48_39.map',sep='\t',header=None)


dd = dict(zip(map_39.iloc[:,0],map_39.iloc[:,1]))


d2 = dict(zip(map_39.iloc[:,1].unique(),range(39)))

data['2_y'] = data['1_y'].map(dd)
data['label_num'] = data['2_y'].map(d2)


train = np.array(data.iloc[:,1:40])

train3 = train[:123*int(len(train)/123)]

train3 = train3.reshape((int(train3.shape[0]/123),123,39))

data_ = []
for i in range(train3.shape[0]):
    if i%500 ==0:
        print (i)
    data_.append([])
    for j in range(train3.shape[1]):
        data_[i].append([])
        #data[i][j].append()
        if j==0:
            data_[i][j].append(train3[i,j].tolist())
            data_[i][j].append(train3[i,j].tolist())
            data_[i][j].append(train3[i,j+1].tolist())
        elif j==train3.shape[1]-1:
            data_[i][j].append(train3[i,j-1].tolist())
            data_[i][j].append(train3[i,j].tolist())
            data_[i][j].append(train3[i,j].tolist())
        else:
            data_[i][j].append(train3[i,j-1].tolist())
            data_[i][j].append(train3[i,j].tolist())
            data_[i][j].append(train3[i,j+1].tolist())
train5= np.array(data_)
train5 = train5.reshape(train5.shape[0],train5.shape[1],train5.shape[2],train5.shape[3],1)
def one_hot(number):
    xx = [0]*39
    xx[int(number)]= 1
    return xx



label = np.array(data['label_num'])

label3 = np.array([one_hot(i) for i in label])

label4 = label3[:123*int(len(label3)/123)]

label5 = label4.reshape((int(label4.shape[0]/123),123,-1))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train5, label5, test_size=0.01)



from keras.models import Sequential
from keras.layers import *
from keras import callbacks


model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 1)), input_shape=(123, 3, 39, 1)))
#model.add(TimeDistributed(Dense(39)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(39)))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

callbacks = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

for i in range(20):
	model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), callbacks=[callbacks])
	#score = model.evaluate(X_test, y_test, batch_size=16)
	val_loss = model.evaluate(X_train, y_train)
	model_name = 'model_CNN_acc_{:8f}'.format(val_loss[1])
	model.save(model_name)





