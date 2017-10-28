

import pandas as pd
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import *

def to_A(num):
    xx =''
    for i in num:
        xx+=chr2A[int2chr[i]]
    return xx

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

test_data = {}

R_size = 0
print('Starting processing testing data.......')
for i in test_raw.index:
    #if i%10000==0:
        #print(i)
    key = test_raw.loc[i,'name']
    if key not in test_data:
        test_data[key] = []
        #test_data[key] = []
    test_data[key].append(list(test_raw.iloc[i,1:40]))
    #test_data[key].append(one_hot(test_raw['label_num'].loc[i]))
#maks=max(test_data, key=lambda k: len(test_data[k]))

R_size = 777
print("R_size:",R_size)
test = []
label = []
sil = [0]*39
sil[7] = 1
print('Starting padding testing data.......')
for key in test_data:
    #test_data[key] +=[[0]*39]*(R_size-len(test_data[key]))
    #train_label[key] += [sil]*(R_size-len(train_data[key]))
    if len(test_data[key]) <= R_size:
        ob = test_data[key]+[[0]*39]*(R_size-len(test_data[key]))
    else:
        ob = test_data[key][:R_size]
    test.append(ob)
    #label.append(train_label[key])


test = np.array(test)




K = model.predict(test)
ans = np.argmax(K, axis=2)
ans2 = np.max(K, axis=2)

# Label = ans.reshape(len(B),1)
# Label = Label[:len(testing)]

# Label2 = ans2.reshape(len(B),1)
# Label2 = Label2[:len(testing)]

print('Starting processing testing data.......')
test_label ={}
count = 0
for key in test_data:
    #test_label[key]
    label_tmp = ans[count,:len(test_data[key])]
    label_tmp2 = ans2[count,:len(test_data[key])]

    label_tmp = [label_tmp[i] for i in range(len(label_tmp)) if label_tmp2[i] >0.75]
    #label_tmp = [label_tmp[i] for i in range(len(label_tmp)) ]
    test_label[key] = to_A(rephrase(label_tmp))
    count+=1


# for i in test_raw.index:
#     key = test_raw.loc[i,'name']
#     if key not in test_data:
#         test_data[key] = []
#     if Label2[i,0] > 0.8:
#         test_data[key].append(Label[i,0])

# predict_label= {}
# for key in test_data:
#     predict_label[key] = to_A(rephrase(test_data[key]))


print('Writing.......')
with open(sys.argv[2],'w') as output:
    output.write('id,phone_sequence')
    count = 0
    for key in test_label:
        output.write('\n')
        output.write(str(key))
        output.write(',')
        output.write(test_label[key])
        count+=1






