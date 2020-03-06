# import libraries
from __future__ import print_function, division

import torch
import torchvision
import torch.nn as nn
from torchvision import  models
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd
import math
from collections import OrderedDict
from workspace_utils import active_session
from data_processing import process_image
import time




def module_build(module_arch,hidden_units):
    """ this function builds a module from a pretrained module 
    
    Input: 1.module architecture [vgg13, resnet18 and alexnet]
           2. Hidden units
          
           
    Return: a pretrained module 
    """
   
    # define model parameter
    
    hidden_units=hidden_units
    output_size=102
    module_classes={}
    

    if module_arch == 'alexnet' :
        
        # load pretrained modules and freze them
        model = torchvision.models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        
        # get the input size from the first layer of the classifier
        input_size=model.classifier[0].in_features
        
        # new classifier
        classifier= nn.Sequential(nn.Linear(input_size, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, output_size),
                                 nn.LogSoftmax(dim=1)
                                )
        
        model.classifier= classifier
        module_classes[module_arch]=model
        module_classes["classifier"]=classifier
        
    elif module_arch ==  'vgg13' :
        
        # load pretrained modules and freze them
        
        model = torchvision.models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        
        # get the input size from the first layer of the classifier
        input_size=model.classifier[0].in_features
        # new classifier
        classifier= nn.Sequential(nn.Linear(input_size, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, output_size),
                                 nn.LogSoftmax(dim=1)
                                )
        
        model.classifier= classifier
        module_classes[module_arch]=model
        module_classes["classifier"]=classifier
        
        
    else:
        
        # load pretrained modules and freze them
        model = torchvision.models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
               
        # get the input size from the linear layer 
        input_size=model.fc.in_features
        # new classifier
        classifier=Linear(input_size,output_size)
        model.fc = classifier
        
        module_classes[module_arch]=model
        module_classes["classifier"]=classifier
   
   



    
    return module_classes ,input_size ,output_size ,hidden_units




def train_module(dataloaders,module,device,arc, learning_rate , epochs):
    """ this function trains the module on the training data
    Input: it takes in 6 parameters
                1. dataloaders
                3. built pretrained module 
                4. device to train on whether cpu or gpu
                5. architecture name
                6. learning rate 
                7. epochs
    Return: prints out Training Loss, Validation loss and Validation accuracy and the final model ,optimizer ,epochs,learning_rate and device 
    
    """
    
    start = time.time()
    model_end=False
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
 
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
        device='cpu'
    else:
        print('CUDA is available!  Training on GPU ...')
        
    
    # geting the model    
    model=module[arc]
    # setting criterion and optimizer for model based on the architecture name

    criterion=  nn.CrossEntropyLoss()
    if  arc == "alexnet" or "v13":
        optimizer=  optim.SGD(model.classifier.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.to(device)
    # set the parameters for training data
    epochs = epochs
    running_loss = 0
    accuracy = 0
    
    # wrap the training in an active section to prevent the work space from staying idle and shutting down
    with active_session():
        
        print("Training has started------------------------------------------------------------------------")
        
        
        for epoch in range(epochs):
            # set the model of the model
            train_mode = 0
            valid_mode = 1
            
            for mode in [train_mode, valid_mode]:   
                if mode == train_mode:
                    model.train()
                else:
                    model.eval()
                pass_count = 0
            
            
                for data in dataloaders[mode]:
                    pass_count += 1
                
                    images, labels = data
                    # set the images and labels to preferred device 
                    images, labels = images.to(device), labels.to(device)
    
                
                    # zero out the weights or gradients to begin training
                    optimizer.zero_grad()
                    # Forward
                    output = model(images)
                    loss = criterion(output, labels)
                    # Backward
                    if mode == train_mode:
                        loss.backward()
                        optimizer.step()                

                    running_loss += loss.item()
                
                    # Calculate  the accuracy of the model
                    ps = torch.exp(output).data
                    
                    equality = (labels.data == ps.max(1)[1])
                    
                    accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
            
                if mode == train_mode:
                    print("\nEpoch: {}/{} ".format(epoch+1, epochs),
                        "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
                else:
                    print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                    "Accuracy: {:.4f}".format(accuracy))
            
                running_loss = 0   

        
    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))          
    print("Model training successfuly finished------------------------------------------------------------------------")  
    model_end=True
    return model ,epochs   ,learning_rate   ,device  ,model_end
                
                
def test_module(testloader,module,device):
    
    
    """ this function trains the module on the training data
    Input: it takes in three parameters
                1. test data
                2. built pretrained module 
                3. device to train on whether cpu or gpu
    Return: prints test accuracy
    
    """
    # define criterion    
    criterion=  nn.CrossEntropyLoss()     
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
 
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
        device='cpu'
    else:
        print('CUDA is available!  Training on GPU ...')
           
    model= module  
    # geting the model    
    model.to(device)
    
    #  zero the test loss at the start of testing
    test_loss = 0.0

    model.eval() # eval mode
    accuracy=0
    # disable training and validate data
    with torch.no_grad():
        
        # iterate over test data
        for images, labels in testloader:
            # set the images and labels to preferred device 
            images, labels = images.to(device), labels.to(device)
    
            # predict the image using the model
            output = model(images)
            # get the loss
            loss = criterion(output, labels)
            test_loss += loss.item()
    
            # Calculate  the accuracy of the model
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    

    return print('Accuracy of the network on the  test images: %d %%' % (100 * accuracy/len(testloader)))




def save_model(module_param ,arc ,save_dir, model_tosave ,train_image_data,epochs ,learning_rate,batch_size,input_size ,output_size ,hidden_size):
    
    """ this function saves the trained model
    Input: it takes in the module,arc,save_dir, model_tosave,trainloader,,epochs ,learning_rate,batch_size
               
    Return: None
    
    """
    # geting the model and its parameters   
    module_classes = module_param
    # choose the classifier from the architecture name
    classifier= module_classes["classifier"]   
        
    model_tosave.class_to_idx = train_image_data.class_to_idx

    # create a checkpoint for saving my model
    checkpoint = {
              'input_size': input_size,
              'output_size': output_size,
              'model_architecture': arc,
              'learning_rate': learning_rate,
              'batch_size': batch_size,
              'classifier' : classifier,
              'epochs': epochs,
              'state_dict': model_tosave.state_dict(),
              'class_to_idx': model_tosave.class_to_idx}    
    
    torch.save(checkpoint,save_dir )
    print("Model successfuly saved")
    
    
    

def load_mycheckpoint(path):
    """ this function loads the saved model
    Input: it takes in the path to the saved checkpoint
               
    Return: the saved model
    
    """
    
    
  
    
    # load checkpoint
    checkpoint = torch.load(path)
    
    # get the model architecture used for training the mode
    model_arc=checkpoint['model_architecture']
    
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, model_arc)(pretrained=True)
    

    
    if  model_arc =="alexnet" or "vgg13":
        model.classifier = checkpoint['classifier']
    else:
        model.fc= checkpoint['classifier']
        
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
 

    print("Model successfuly loaded")
    return model


def predict(image_path, model, topk,device):
    
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    """
    
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
 
    if not train_on_gpu:
        print('CUDA is not available.  Predicting on CPU ...')
        device='cpu'
    else:
        print('CUDA is available!  Predicting on GPU ...')
    
    
    # process the image
    image = process_image(image_path)

    
    # fourth dimension to the beginning to indicate batch size
    image = image[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(image)
    image = image.float()
    
    
    image_input=image.to(device)
    model.to(device)
    # enter image through our model
    out = model(image_input)
    
    # Reverse the log function in our output
    out = torch.exp(out)
    
    # Get the top predicted class, and the output percentage for that class
    probs, classes = out.topk(5)


    return probs.cpu().data.numpy(), classes.cpu().data.numpy()
    
   



   
    
    
    

 




    