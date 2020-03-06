
# PROGRAMMER: Ian Akoto
# DATE CREATED:   9/02/2020

# Imports python modules
import argparse


def get_input_args():
    """
    Retrieves and parses the  command line arguments provided by the user when
    they run the program from a terminal window. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'data_dir'
      2. Model Architecture as --arch with default value 'vgg'
      3. Device to train on default cpu
      4. Learning rate default value is 0.001
      5. Number of Epochs for training default value is 2
      6. Number of Hidden units default value is 528
      7. Save directory for checkpoint default is 'saved_checkpoint.pth'
      8. Top k values for prediction default value is  3
      9.Category names for mapping category to names default value cat_to_name.json
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Retrieves and parses the 3 command line arguments')
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('dir',type=str, default='ImageClassifier\flowers', help='get data directory from command line')
    parser.add_argument('--arch',type=str, default='vgg13', help='Model Architecture as --arch with  vgg13 and resnet18 and alexnet default is vgg13')
    parser.add_argument('--gpu',type=str, nargs='?',const='cuda:0',  default='cuda:0', help='Device to train on default gpu')
    parser.add_argument('--learning_rate',type=int, default=0.01, help='Learning rate default value is 0.001')
    parser.add_argument('--epochs',type=int, default=8 , help=' Number of Epochs for training default value is 2')
    parser.add_argument('--hidden_units',type=int, default=1528, help='Number of Hidden units default value is 528')
    parser.add_argument('--save_dir',type=str, default='saved_checkpoint.pth', help='Save directory for checkpoint default is saved_checkpoint.pth')
    parser.add_argument('--top_k',type=int, default=3, help='Top k values for prediction default value is  3')
    parser.add_argument('--category_names',type=str, default='', help='json file map category to names default value cat_to_name.json')
    
    parser.add_argument('checkpoint', nargs='?', action="store",type=str)

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()
