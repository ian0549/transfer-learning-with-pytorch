
# PROGRAMMER: Ian Akoto
# DATE CREATED:   9/02/2020
# Usage: sh run_prediction.sh    -- will run program from commandline within Project Workspace
#  Add other parameters such as checkpoint --top_k 3





python predict.py  flowers/test/10/image_07117.jpg  saved_checkpoint.pth  --gpu --category_names cat_to_name.json