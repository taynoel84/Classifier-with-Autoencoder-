# Classifier-with-Autoencoder-
A study in applying autoencoder for classification task

## Implementation
### Require
PyTorch
### Data Preparation
Use/modify datasetPreparation.py to prepare training and testing dataset. This code uses CIFAR-10 [dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (Python Version). Set DATASET_DIR and TARGET_SAVE_DIR variable to source and target directory respectively.

### Implementation
main.py is the main implementation code. Following are the arguments:
```
"--trainImL" : training image file location (prepared by datasetPreparation.py)
"--trainLabL" : training label file location (prepared by datasetPreparation.py)
"--testImL" : testing image file location (prepared by datasetPreparation.py)
"--testLabL" : testing label file location (prepared by datasetPreparation.py)
"--batchSize" : Training batch size
"--epochNum" : Number of training epochs
"--pretrainedFile" : Location of pretrained parameter file
"--imageSize" : Image size (currently only allows 32)
"--classifierInpSize" : Input vector size to classifier network
"--classificationLossWeight" : Currently set to 1

"--denoisingAutoEncoder" : Option to use autoencoder during training or not
"--reconstructionnLossWeight" : Weight for image reconstruction loss

"--useVariational" : Option to use sampling or not for classifier input
"-KLLossWeight" : Weight for KL divergence loss

"--useSpatialTransform" : Option to use Spatial Transformer
```
