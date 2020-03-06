
# PROGRAMMER: Ian Akoto
# DATE CREATED:   9/02/2020  

#import functions
from module_functions import load_mycheckpoint
from module_functions import predict
from data_processing import process_image
from get_input_args import get_input_args
import  sys
import pandas as pd
import json
import math


in_arg = get_input_args()
# get the values from the command line for prediction 
cat_to_name=in_arg.category_names
gpu = in_arg.gpu
top_k=in_arg.top_k



# Get the number of arguments
numArgs = len(sys.argv)





if numArgs >= 3:
    # Retrieve the input directory
    image_file = sys.argv[1]
    # Retrieve the output file
    check_point = sys.argv[2]
    
 
    # load model
    model= load_mycheckpoint(check_point)


    # predict takes in image_path, model, topk=3,device
    probs, classes=predict(image_file,model,top_k,gpu)
     
    if  cat_to_name!=''  :
        
        # open category file
        with open(cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
        # map the classes number to the actual names
        class_to_name=list(map(lambda x: cat_to_name[str(x)], classes[0].tolist()))

        # map the probabilities  to percentages
        prob_to_percent=list(map(lambda x:math.ceil( x*100), probs[0].tolist()))
        
        predict_data=  pd.DataFrame.from_dict(dict([('class', class_to_name), ('prob', prob_to_percent)]))
        print(predict_data)
    else:
        # map the classes number to the actual names
        classes= classes[0].tolist()

        # map the probabilities  to percentages
        prob=list(map(lambda x: math.ceil( x*100), probs[0].tolist()))
        
        predict_data=  pd.DataFrame.from_dict(dict([('class', classes), ('prob', prob)]))
        
        print("*************Prediction Results****************")
        print(predict_data)
        
    
else:
    print("ERROR. Command should have the form: python predict.py /path/to/image checkpoint")

