import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import random

"""
For reproducibility
"""
from random import seed as rndSeed
#rndSeed(5)
#torch.manual_seed(0)

class ListDataset(Dataset):
    def __init__(self,imSize,imLFile,labLFile,setRandom=False):
        self.imSize=imSize
        self.imL=np.load(imLFile)
        self.labL=np.load(labLFile)
        self.setRandom=setRandom
    
    def __len__(self):
        return self.imL.shape[0]
    
    def __getitem__(self,index):
        imL=torch.from_numpy(self.imL[index].astype(np.float32))
        labL=self.labL[index]
        
        if self.setRandom:
            #Random zoom and shift
            _dS=random.choice(range(6))
            _x=0 if _dS==0 else random.choice(range(_dS))
            _y=0 if _dS==0 else random.choice(range(_dS))
            newImSize=self.imSize-_dS
            imL=imL[:,_y:_y+newImSize,_x:_x+newImSize]
            imL=F.interpolate(imL.unsqueeze(0),size=self.imSize,mode="nearest")[0]
            
            #Random flip
            if random.choice([0,1]):
                imL=imL.flip(2)
        
        return imL,labL 
    
    def collate_fn(self,batch):
        imL,labL=list(zip(*batch))
        imL=torch.stack(imL,dim=0)
        labL=torch.from_numpy(np.array(labL))
        return imL,labL
     
   
        
        