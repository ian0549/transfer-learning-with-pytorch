

# PROGRAMMER: Ian Akoto
# DATE CREATED:   9/02/2020

#import functions

import  sys
from module_functions import module_build
from module_functions import train_module
from module_functions import test_module
from module_functions import save_model

from data_processing import load_process_data

from get_input_args import get_input_args


 
def main():
    """ Define function as main to run when called"""
    
    print("************************************ IMAGE CLASSIFIER TRAINING, TESTING AND SAVING APP****************************************"   )  
    #  Define get_input_args function within the file get_input_args.py
    in_arg = get_input_args()
    

    
    # get the values from the command line for training 
    data_dir =sys.argv[1]
    
    model_architecture = in_arg.arch
    learning_rate = in_arg.learning_rate
    epochs = in_arg.epochs
    hidden_units = in_arg.hidden_units
    save_dir = in_arg.save_dir
    gpu =in_arg.gpu

    
    
    # get the dataset 
    # this function returns training, testing and validation dataset
    image_datasets,dataloaders,batch_size  =load_process_data(data_dir)
    
    # build model
    # it takes in module_arch,hidden_units
    # and returns model, input size, output size and hidden size
    module_built,input_size ,output_size ,hidden_size =module_build(model_architecture,hidden_units)
    
    
    
    
    # train the module 
#    it takes in 7 parameters
#                 1. training data
#                 2. validation data
#                 3. built pretrained module 
#                 4. device to train on whether cpu or gpu
#                 5. architecture name
#                 6. learning rate 
#                 7. epochs
    model,epochs ,learning_rate ,device,model_end= train_module(dataloaders,
                                                                            module_built,
                                                                            gpu,
                                                                            model_architecture,
                                                                            learning_rate,
                                                                            epochs)
    
    # test the model
    
#    it takes in three parameters
#                 1. test data, dataloaders[2]
#                 2. built pretrained module 
#                 3. device to train on whether cpu or gpu
#     Return: prints test accuracy

    # check to see if model has finished training and test it
    
    if model_end:
        test_module(dataloaders[2],model,device)
    
    
    # save the mode
    # Input: it takes in the module_param,arc,save_dir, model_tosave,device,trainloader,,epochs ,learning_rate,batch_size
    #
    # check to see if model has finished training and save it
    if model_end:
        save_model(module_built,
                   model_architecture,
                   save_dir,model,
                   image_datasets[0],
                   epochs,
                   learning_rate,
                   batch_size,
                   input_size ,
                   output_size,
                   hidden_size )
    
    
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    