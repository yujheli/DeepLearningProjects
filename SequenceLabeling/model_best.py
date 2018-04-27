
# coding: utf-8

#from scikits.statsmodels.tools import categorical
#b = categorical(a, drop=True)
#keras.utils.to_categorical(y, num_classes=39)

import pandas as pd
import numpy as np
import sys


def one_hot(number):
    xx = [0]*39
    xx[int(number)]= 1
    return xx

raw = pd.read_csv(sys.argv[1]+'/mfcc/train.ark',sep=' ',header=None)

label = pd.read_csv(sys.argv[1]+'/label/train.lab',header=None)


data = pd.merge(raw, label,how='left',on=[0])#.iloc[:,-1]

map_39 = pd.read_csv(sys.argv[1]+'/phones/48_39.map',sep='\t',header=None)


dd = dict(zip(map_39.iloc[:,0],map_39.iloc[:,1]))


d2 = dict(zip(map_39.iloc[:,1].unique(),range(39)))

data['2_y'] = data['1_y'].map(dd)
data['label_num'] = data['2_y'].map(d2)
def to_name(strr):
    return '_'.join(strr.split('_')[:2])

data['name'] = data[0].apply(to_name)

train_data = {}
train_label = {}
R_size = 0
print('Starting processing training data.......')
for i in data.index:
    if i%10000==9999:
        #break
        print(i)
    key = data.loc[i,'name']
    if key not in train_data:
        train_data[key] = []
        train_label[key] = []
    train_data[key].append(list(data.iloc[i,1:40]))
    train_label[key].append(list(one_hot(data['label_num'].loc[i])))
maks=max(train_data, key=lambda k: len(train_data[k]))

R_size = len(train_data[maks])
print("R_size:",R_size)
train = []
label = []
sil = [0]*39
sil[7] = 1
print('Starting padding training data.......')
for key in train_data:
    train_data[key] +=[[0]*39]*(R_size-len(train_data[key]))
    train_label[key] += [sil]*(R_size-len(train_label[key]))
    train.append(train_data[key])
    label.append(train_label[key])


train = np.array(train)
print train.shape
# train3 = train[:123*int(len(train)/123)]

# train3 = train3.reshape((int(train3.shape[0]/123),123,39))

label = np.array(label)
print label.shape
# label3 = np.array([one_hot(i) for i in label])

# label4 = label3[:123*int(len(label3)/123)]

# label5 = label4.reshape((int(label4.shape[0]/123),123,-1))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.05)



from keras.models import Sequential
from keras.layers import *
from keras import callbacks

model = Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True),
            input_shape=(R_size, 39)))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(39)))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

callbacks = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
for i in range(20):
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[callbacks])
    #score = model.evaluate(X_test, y_test, batch_size=16)
    val_loss = model.evaluate(X_test, y_test)
    model_name = sys.argv[1]+'/model_Pad_easy_acc_{:8f}'.format(val_loss[1])
    model.save(model_name)




