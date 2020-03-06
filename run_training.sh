
# PROGRAMMER: Ian Akoto
# DATE CREATED:   9/02/2020
# Usage: sh run_training.sh    -- will run program from commandline within Project Workspace
#  Add other parameters such as epoch, learning rate and others
python train.py ImageClassifier/flowers --gpu --category_names cat_to_name.json

