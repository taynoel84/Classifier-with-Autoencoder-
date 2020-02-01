import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualModule(nn.Module):
    def __init__(self,chanNum):
        super(ResidualModule,self).__init__()
        pad=nn.ReplicationPad2d((1,1,1,1))
        self.conv=nn.Sequential(pad,nn.Conv2d(chanNum,chanNum,3,stride=1,padding=0,bias=False),nn.BatchNorm2d(chanNum),nn.LeakyReLU(),
                            pad,nn.Conv2d(chanNum,chanNum,3,stride=1,padding=0,bias=False),nn.BatchNorm2d(chanNum),nn.LeakyReLU())
    
    def forward(self,xIn):
        return self.conv(xIn)+xIn

class Encoder(nn.Module):
    def __init__(self,opt):
        super(Encoder,self).__init__()
        self.useSpatialTransform=opt.useSpatialTransform
        self.classifierInpSize=opt.classifierInpSize
        pad=nn.ReplicationPad2d((1,1,1,1))
        
        conv0=nn.Sequential(pad,nn.Conv2d(3,32,3,stride=1,padding=0,bias=False), nn.BatchNorm2d(32), nn.LeakyReLU())
        conv1=nn.Sequential(pad,nn.Conv2d(32,64,3,stride=1,padding=0,bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),nn.MaxPool2d(2,2))#32->16
        conv1_r=ResidualModule(64)
        conv2=nn.Sequential(pad,nn.Conv2d(64,128,3,stride=1,padding=0,bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(),nn.MaxPool2d(2,2))#16->8
        conv3=nn.Sequential(pad,nn.Conv2d(128,64,3,stride=1,padding=0,bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),nn.MaxPool2d(2,2))#8->4
        conv3_r=ResidualModule(64)
        self.convVar=nn.Sequential(pad,nn.Conv2d(64,self.classifierInpSize,3,stride=1,padding=0))
        
        self.fullConv1=nn.Sequential(conv0,conv1,conv1_r)
        self.fullConv2=nn.Sequential(conv2,conv3,conv3_r)
        
        
        convSTN0=nn.Sequential(nn.Conv2d(64,64,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(64), nn.LeakyReLU())#16->8
        convSTN1=nn.Sequential(nn.Conv2d(64,32,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(32), nn.LeakyReLU())#8->4
        convSTN2=nn.Sequential(nn.Conv2d(32,32,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(32), nn.LeakyReLU())#4->2
        convSTN3=nn.Sequential(nn.Conv2d(32,6,3,stride=2,padding=1))#2->1
        self.fullConvSTN=nn.Sequential(convSTN0,convSTN1,convSTN2,convSTN3)
        
    
    def forward(self,x):
        x=self.fullConv1(x)
        
        if self.useSpatialTransform:
            tranMat=self.fullConvSTN(x).view(-1,2,3)
            grid=F.affine_grid(tranMat,x.size())
            x=F.grid_sample(x,grid)
        
        
        x=self.fullConv2(x)
        logVar=self.convVar(x)
        return x,logVar
        
class Decoder(nn.Module):
    def __init__(self,opt):
        super(Decoder,self).__init__()
        conv0=nn.Sequential(nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=1,bias=False), nn.BatchNorm2d(64),nn.LeakyReLU())#4->8
        conv1=nn.Sequential(nn.Conv2d(64,64,3,stride=1,padding=1,bias=False), nn.BatchNorm2d(64), nn.LeakyReLU())
        conv2=nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.LeakyReLU())#8->16
        conv3=nn.Sequential(nn.ConvTranspose2d(32,3,3,stride=2,padding=1,output_padding=1))#16->32
        self.fullConv=nn.Sequential(conv0,conv1,conv2,conv3)
    
    def forward(self,x):
        return torch.sigmoid(self.fullConv(x))*2-1

class Classifier(nn.Module):
    def __init__(self,opt):
        super(Classifier,self).__init__()
        self.fc=nn.Sequential(nn.Linear(opt.classifierInpSize,24),nn.LeakyReLU(),
                                       nn.Linear(24,10))
    
    def forward(self,x):
        return self.fc(x)

class AutoEncClassifier(nn.Module):
    def __init__(self,opt):
        super(AutoEncClassifier,self).__init__()
        self.encoder=Encoder(opt)
        self.decoder=Decoder(opt)
        self.classifier=Classifier(opt)
        self.classifierInpSize=opt.classifierInpSize
        
        self.useVariational=opt.useVariational
        self.classificationLossWeight=opt.classificationLossWeight
        self.reconstructionnLossWeight=opt.reconstructionnLossWeight
        self.KLLossWeight=opt.KLLossWeight
    
    def loadPretrainedParams(self,paramFile):
        deviceBool=next(self.parameters()).is_cuda
        device=torch.device("cuda:0" if deviceBool else "cpu")
        try:
            pretrainedDict=torch.load(paramFile,map_location=device.type)
            modelDict=self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict }
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except:
            print("Can't load pre-trained parameter files")
            
    def normSampling(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    
    def forward(self,xIn,target=None,VAETrain=False,SELSUP=False):
        #encoding
        if SELSUP:
            xIn2=xIn.clone()
            hRan=(torch.rand(250)*31).type(torch.LongTensor).to(xIn2.device)
            wRan=(torch.rand(250)*31).type(torch.LongTensor).to(xIn2.device)
            xIn2[:,:,hRan,wRan]*=0
            self.debug1=xIn2
            zMu,zLogVar=self.encoder(xIn2)
        else:
            zMu,zLogVar=self.encoder(xIn)
        
        #decoding
        xDec=self.decoder(zMu)
        
        #sampling
        batchNum,chanNum,hNum,wNum=zMu.shape
        xMu=nn.MaxPool2d(4)(zMu).view(batchNum,-1)[:,:self.classifierInpSize]
        xLogVar=nn.MaxPool2d(4)(zLogVar).view(batchNum,-1)[:,:self.classifierInpSize]
        if self.training and self.useVariational:
            xEnc=self.normSampling(xMu,xLogVar)
        else:
            xEnc=xMu
        
        
        
        #classification part
        xLinear=self.classifier(xEnc)
        
        
        #//////////////////////////////////////////////////////////////////////////////////////////////
        #                       Loss calculation
        #//////////////////////////////////////////////////////////////////////////////////////////////
        #classification loss
        if target is not None:
            ceLoss=F.cross_entropy(xLinear,target)
        else:
            ceLoss=torch.tensor(0)
        
        
        #VAE loss (reconstruction and KL loss)
        if VAETrain:
            batchNum,chanNum,hNum,wNum=xDec.shape
            #reconLoss=F.binary_cross_entropy(xDec.view(batchNum,-1), xIn.view(batchNum,-1), reduction='sum')
            reconLoss=F.mse_loss(xDec.view(batchNum,-1), xIn.view(batchNum,-1),reduction='mean')
            KLLoss=-0.5*torch.mean(1+xLogVar-xMu.pow(2)- xLogVar.exp())
            #print("%.3f,  %.3f,  %.3f"%(ceLoss.item(),reconLoss.item(),KLLoss.item()))
        else:
            reconLoss=torch.tensor(0)
            KLLoss=torch.tensor(0)

        if SELSUP:
            loss=self.reconstructionnLossWeight*reconLoss
        else:
            loss=self.classificationLossWeight*ceLoss+self.KLLossWeight*KLLoss
        
        self.debug2=xDec
        return xLinear,xDec,loss,[ceLoss.cpu(),reconLoss.cpu(),KLLoss.cpu()]



