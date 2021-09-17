import argparse
import logging
from src.preprocessing import *

# create logging configs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# create parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='data/')
args = parser.parse_args()

main_folder = args.input_dir

# user needs to input these
# Label ids of the dataset
category_ids = {
    "labo": 0
}

# Define which colors match which categories in the images
# multiply value in json by 255
category_colors = {
    "(0, 0, 255)": 0, # labo
}

# Define the ids that are a multiplolygon. e.g. wall, roof and sky
multipolygon_ids = []



# run the function to rename all files to make them unique
logging.info(f'---Start renaming all files---/n')
unique_files(main_folder)
logging.info(f'---Finished renaming all files---/n/n')

# run the function to copy all files to a single directory
logging.info(f'---Start copying all files---/n')
group_data(main_folder)
logging.info(f'---Finished copying all files---/n/n')

# do train and validation split on all images
logging.info(f'---Start creating training and validation folders---/n')
train_val_split(main_folder)
logging.info(f'---Finished train-val split---/n/n')


# perform extra image augmentations with albumentation library
logging.info(f'---Start generating image augmentations---/n')
img_augmentation(main_folder)
logging.info(f'---Finished generating image augmentations---/n/n')


# create COCO JSON files for each train and val set
train_dir = os.path.join(main_folder, 'final/train')
val_dir = os.path.join(main_folder, 'final/val')

logging.info(f'---Start creating COCO JSON annotations---/n')
coco_pipeline(train_dir, category_ids, category_colors, multipolygon_ids)
logging.info(f'Completed for training set!')
coco_pipeline(val_dir, category_ids, category_colors, multipolygon_ids)
logging.info(f'Completed for validation set!/n/n')

