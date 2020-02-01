import os
import torch
from torch.autograd import Variable
import torch.optim as optim

from dataset import ListDataset
from torch.utils.data import DataLoader
import argparse 
import model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainImL', type=str, default='E:/Datasets/cifar-10-batches-py/prepDat/trainIm.npy',
                        help='Training image dataset file')
    parser.add_argument('--trainLabL', type=str, default='E:/Datasets/cifar-10-batches-py/prepDat/trainLab.npy',
                        help='Training label file')
    parser.add_argument('--testImL', type=str, default='E:/Datasets/cifar-10-batches-py/prepDat/testIm.npy',
                        help='Testing image dataset file')
    parser.add_argument('--testLabL', type=str, default='E:/Datasets/cifar-10-batches-py/prepDat/testLab.npy',
                        help='Testing label file')
    
    parser.add_argument('--batchSize', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--epochNum', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--pretrainedFile', type=str, default='param.dict',
                        help='Pretrained parameter file')
    
    
    parser.add_argument('--imageSize', type=int, default=32,
                        help='Image size')
    parser.add_argument('--classifierInpSize', type=int, default=24,
                        help='Classifier input size')
    parser.add_argument('--classificationLossWeight', type=float, default=1.0,
                        help='Weight for classification loss')
    
    parser.add_argument('--reconstructionnLossWeight', type=float, default=00.0,
                        help='Weight for image reconstruction loss')
    parser.add_argument('--denoisingAutoEncoder', type=bool, default=False,
                        help='Option to train denoising autoencoder')
    
    
    parser.add_argument('--KLLossWeight', type=float, default=0.00,
                        help='Weight for KL-divergence loss')
    parser.add_argument('--useVariational', type=bool, default=False,
                        help='Enable encoder output sampling')
    parser.add_argument('--useSpatialTransform', type=bool, default=False,
                        help='Enable spatial transformer')
    
    
    args = parser.parse_args()
    return args


class Optimizer(object):
    def __init__(self,net):
        self.optimizerAdam=optim.Adam(net.parameters(),lr=0.001,eps=1e-4)
        self.optimizerSGD=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)
        self._optim=self.optimizerAdam
        self.SGD_LR=[0.04,0.04,0.02]
        self.Adam_MaxEpoch=7
    def __call__(self,epoch):
        if epoch<self.Adam_MaxEpoch:
            self._optim=self.optimizerAdam
        else:
            for param_group in self.optimizerSGD.param_groups:
                param_group['lr']=self.SGD_LR[min(epoch-self.Adam_MaxEpoch,len(self.SGD_LR)-1)]
            self._optim=self.optimizerSGD
        self._optim.step()
    def zero_grad(self):
        self._optim.zero_grad()
        

def imshow(imT):
    #Input: Torch tensor with shape (channel, height, wdith)
    import matplotlib.pyplot as plt
    imMean=np.array(CIFAR_MEAN).reshape(1,1,3)
    im=imT.numpy().transpose(1,2,0)+imMean
    plt.imshow(im)


torch.backends.cudnn.benchmark=True
CIFAR_LABEL={0:"airplane", 1:"automobile", 2:"bird",3:"cat",4:"deer",
                        5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
CIFAR_MEAN=[125.3/255, 123.0/255, 113.9/255]

if __name__=='__main__':
     options=parse_args()
     device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
     net=model.AutoEncClassifier(options)
     net.to(device)
     trainDataset=ListDataset(options.imageSize,options.trainImL,options.trainLabL,setRandom=True)
     testDataset=ListDataset(options.imageSize,options.testImL,options.testLabL)
     
     trainDataloader= DataLoader(
                 trainDataset,
                 batch_size=options.batchSize,
                 shuffle=True,
                 num_workers=2,
                 pin_memory=True,
                 collate_fn=trainDataset.collate_fn)
     
     testDataloader= DataLoader(
                 testDataset,
                 batch_size=200,
                 shuffle=False,
                 num_workers=2,
                 pin_memory=True,
                 collate_fn=testDataset.collate_fn)
     
     net.loadPretrainedParams(options.pretrainedFile)
     optimizer= Optimizer(net)
     
     for fEpoch in range(options.epochNum):
         print("==========Epoch %d========="%fEpoch)
         
         #Training loop
         lossList=[]
         reconLossList=[]
         net.train()
         for inputs,labels in trainDataloader:
             inputs=Variable(inputs).to(device)
             labels=Variable(labels).type(torch.LongTensor).to(device)
             optimizer.zero_grad()
             
             if options.denoisingAutoEncoder:
                 _,_,loss,lossL=net(inputs,target=labels,VAETrain=True,SELSUP=True)
                 loss.backward()
                 reconLoss=loss.item()
                 reconLossList.append(reconLoss)
             
             
             _,_,loss,lossL=net(inputs,target=labels,VAETrain=True)
             lossList.append(lossL)
             loss.backward()
             optimizer(fEpoch)
         _L=torch.mean(torch.Tensor(lossList),dim=0).tolist()
         if len(reconLossList)>0:
             reconLossMean=torch.mean(torch.Tensor(reconLossList))
             print("Train Loss: %.3f %.3f"%(_L[0],reconLossMean))
         else:
             print("Train Loss: %.3f "%_L[0])
         
        
         #Evaluation loop
         lossList=[]
         net.eval()
         confusionMat=torch.zeros(10,10).type(torch.LongTensor)
         for inputs,labels in testDataloader:
             inputs=Variable(inputs).to(device)
             labels=Variable(labels).type(torch.LongTensor).to(device)
             with torch.no_grad():
                 xLinear,_,_,lossL=net(inputs,labels,VAETrain=True)
             lossList.append(lossL)
             prediction=xLinear.argmax(dim=1)
             for ind,val in enumerate(prediction):
                 confusionMat[labels[ind],val]+=1
         _L=torch.mean(torch.Tensor(lossList),dim=0).tolist()
         print("Eval Loss:[ %.3f, %.3f, %.3f ]"%(_L[0],_L[1],_L[2]))
         recall=torch.zeros(1)
         precision=torch.zeros(1)
         accuracyList=[]
         for fLab in range(10):
             TP=(confusionMat[fLab,fLab]).type(torch.float32)
             FN=(torch.sum(confusionMat[fLab,:])-TP).type(torch.float32)
             FP=(torch.sum(confusionMat[:,fLab])-TP).type(torch.float32)
             recall+=TP/(TP+FN)
             precision+=TP/(TP+FP)
             accuracyList.append(TP/torch.sum(confusionMat[fLab,:]).type(torch.float32))
         print("Precision: %.2f    Recall: %.2f"%(precision/10,recall/10))
         #for f in range(10):
         #    print("Accuracy for %s is %.2f%%"%(CIFAR_LABEL[f],accuracyList[f]*100))  
         
        
     torch.save(net.state_dict(),"./paramSave/param_backup.dict")
             
          
     
     
       
             
                 
     
         
         
      
     
     
     
     
     
     