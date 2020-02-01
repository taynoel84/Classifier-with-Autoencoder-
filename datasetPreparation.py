import os
import numpy as np
from functools import reduce
import random

DATASET_DIR='D:/dataset/cifar-10-batches-py'
TARGET_SAVE_DIR='D:/dataset/cifar-10-batches-py/prepDat'
TRAIN_TEST_RATIO=[0.8, 0.8, 0.5, 0.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5]
CIFAR_MEAN=[125.3, 123.0, 113.9]
CIFAR_LABEL={0:"airplane", 1:"automobile", 2:"bird",3:"cat",4:"deer",
                        5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

def unpickle(file):
    import pickle
    with open(file,'rb') as fil:
        dic=pickle.load(fil,encoding='bytes')
    return dic

def createDataAndLabel(imList,indexList):
    imL=[np.stack([ imList[f]  for f in fLabIn],axis=0)  for fLabIn in indexList]
    LabelL=[[ind]*val.shape[0] for ind,val in enumerate(imL)]
    LabelL=reduce((lambda x,y:x+y),LabelL)
    
    imL=np.concatenate(imL,axis=0)
    LabelL=np.array(LabelL)
    return imL,LabelL

trainDatasetFileL=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
testDatasetFileL=['test_batch']
random.seed(0)

#load original data
imL,labelL=list(zip(*[[f2[b'data'],f2[b'labels']] for f2 in
    [unpickle(os.path.join(DATASET_DIR,f)) for f in trainDatasetFileL]]))
imL=np.concatenate(imL,axis=0)
labelL=reduce((lambda x,y:x+y),labelL)

#image  pre-processing
imMean=np.array(CIFAR_MEAN).reshape(1,3,1,1)
imL=imL.reshape(imL.shape[0],3,32,32)
imNormL=(imL.astype(np.float32)-imMean)/255

#random train/test dataset splitting
labelIndexL=[[ind for ind,val in enumerate(labelL) if val==fLab] for fLab in range(10)]
trainTestIndL=[]
for fInd,fLabIn in enumerate(labelIndexL):
    sepPt=int(TRAIN_TEST_RATIO[fInd]*len(fLabIn))
    random.shuffle(fLabIn)
    trainTestIndL.append([fLabIn[:sepPt],fLabIn[sepPt:]])
trainIndL,testIndL=list(zip(*trainTestIndL))
trainIndL=trainIndL

#dataset preparation
trainImL,trainLabL=createDataAndLabel(imNormL,trainIndL)
testImL,testLabL=createDataAndLabel(imNormL,testIndL)

#save prepared dataset
np.save("%s/trainIm.npy"%(TARGET_SAVE_DIR),trainImL)
np.save("%s/trainLab.npy"%(TARGET_SAVE_DIR),trainLabL)
np.save("%s/testIm.npy"%(TARGET_SAVE_DIR),testImL)
np.save("%s/testLab.npy"%(TARGET_SAVE_DIR),testLabL)

