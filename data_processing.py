# PROGRAMMER: Ian Akoto
# DATE CREATED:   9/02/2020


# import libraries for loading and processing data
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


def load_process_data(data_dir):
    """ a function to load data from file path and pre-process the images in the file
    
    Input: file path
    
    Return: preprocessed  training, testing and validating  datasets and batch_size
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #define batch size 
    batch_size=64

    # set data transformation for training data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    # set data transformation for testing and validating  data
    test_valid_transforms= transforms.Compose([
                                       transforms.Resize( 256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)    
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    validloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)
    
    image_datasets = [train_data, valid_data, test_data]
    dataloaders = [trainloader, validloader, testloader]
    
    return image_datasets,dataloaders,batch_size 



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # open the image using Image from PIL
    image= Image.open(image) 
      
    # Get the dimensions of the image
    width, height = image.size
    
    # Resize by keeping the aspect ratio, but changing the dimension so the shortest size is 256px
    image = image.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    
    # Get the derived dimensions of the new image size
    width, height = image.size
    
    # the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
 
    image=np.array(image)
   
    
    # Make all values between 0 and 1
    image = image/255
    
    # normalize the image by subtracting mean and dividing by standard deviation
    image = (image - mean)/std
    
   # print(image.shape)
   
  
    # Make the color channel dimension first instead of last
    image = image.transpose((2, 0, 1))
    
    #print(image.shape)
    
 
    return image
   
  
