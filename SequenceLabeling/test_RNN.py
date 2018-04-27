

import pandas as pd
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import *

    

def rephrase(ans):
    xx = []
    key = 7
    for i in ans:
        if i != key:
            key = i
            xx.append(key)
    if xx[-1]==7:
        return xx[:-1]
    return xx


def to_name(strr):
    return '_'.join(strr.split('_')[:2])

model = load_model(sys.argv[3])
model.summary()

test_raw = pd.read_csv(sys.argv[1]+'/mfcc/test.ark',sep=' ',header=None)
map_39 = pd.read_csv(sys.argv[1]+'/phones/48_39.map',sep='\t',header=None)
int2chr = dict(zip(range(39),map_39.iloc[:,1].unique()))

map_A = pd.read_csv(sys.argv[1]+'/48phone_char.map',sep='\t',header=None)
chr2A = dict(zip(map_A.iloc[:,0],map_A.iloc[:,2]))

#sample = pd.read_csv('sample.csv') 


test_raw['name'] = test_raw[0].apply(to_name)

def to_A(num):
    xx =''
    for i in num:
        xx+=chr2A[int2chr[i]]
    return xx



print('Starting using model.......')
testing = np.array(test_raw.iloc[:,1:40])
remain = len(testing)%123
B = np.concatenate((testing,np.zeros(((123-remain),39))))
BB = B.reshape(int(len(B)/123),123,39)
K = model.predict(BB)
ans = np.argmax(K, axis=2)
ans2 = np.max(K, axis=2)

Label = ans.reshape(len(B),1)
Label = Label[:len(testing)]

Label2 = ans2.reshape(len(B),1)
Label2 = Label2[:len(testing)]

test_data ={}
print('Starting processing testing data.......')
for i in test_raw.index:
    key = test_raw.loc[i,'name']
    if key not in test_data:
        test_data[key] = []
    if Label2[i,0] > 0.78:
        test_data[key].append(Label[i,0])

predict_label= {}
for key in test_data:
    predict_label[key] = to_A(rephrase(test_data[key]))

print('Writing.......')
with open(sys.argv[2],'w') as output:
    output.write('id,phone_sequence')
    count = 0
    for key in predict_label:
        output.write('\n')
        output.write(str(key))
        output.write(',')
        output.write(predict_label[key])
        count+=1






