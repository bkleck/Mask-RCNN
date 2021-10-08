import argparse
import logging
import os
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
parser.add_argument('--test_dir', default='data/')
args = parser.parse_args()

main_folder = os.path.join(str(os.getcwd()), args.input_dir)

# fill up the dictionaries with the JSON file generated from Unity
category_ids, category_colors = extract_json(main_folder)

# Define the ids that are a multiplolygon. e.g. wall, roof and sky
multipolygon_ids = []

obj = str(input("What object is this: "))

# run the function to rename all files to make them unique
logging.info('---Start renaming all files---\n')
unique_files(main_folder)
logging.info('---Finished renaming all files---\n\n')

# run the function to copy all files to a single directory
logging.info('---Start copying all files---\n')
group_data(main_folder)
logging.info('---Finished copying all files---\n\n')

# do train and validation split on all images
logging.info('---Start creating training and validation folders---\n')
train_val_split(main_folder)
logging.info('---Finished train-val split---\n\n')


# perform extra image augmentations with albumentation library
logging.info('---Start generating image augmentations---\n')
img_augmentation(main_folder)
logging.info('---Finished generating image augmentations---\n\n')


# create COCO JSON files for each train and val set
train_dir = os.path.join(main_folder, 'final/train')
val_dir = os.path.join(main_folder, 'final/val')

logging.info('---Start creating COCO JSON annotations---\n')
coco_pipeline(train_dir, category_ids, category_colors, multipolygon_ids, obj)
logging.info('Completed for training set!')
coco_pipeline(val_dir, category_ids, category_colors, multipolygon_ids, obj)
logging.info('Completed for validation set!\n\n')


# create our test dataset suitable for input into Detectron2 database
# only for images for now
test_folder = os.path.join(str(os.getcwd()), args.test_dir)
test_folder = os.path.join(test_folder, 'images')
test_coco(test_folder, category_ids, obj)
logging.info('Completed for test set!\n\n')