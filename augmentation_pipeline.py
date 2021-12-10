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
parser.add_argument('--augmentation', default='False')
parser.add_argument('--real_dir', default='False')
args = parser.parse_args()

main_folder = os.path.join(str(os.getcwd()), args.input_dir)

# get nested folder as well
inner_folder = os.listdir(main_folder)[0]
inner_path = os.path.join(main_folder, inner_folder)

# fill up the dictionaries with the JSON file generated from Unity
category_ids, category_colors, count, object_of_interest, object_id = extract_json(main_folder)

# Define the ids that are a multiplolygon. e.g. wall, roof and sky
multipolygon_ids = []

# get the current object we are looking at
obj = object_of_interest
object_id = object_id

# run the function to rename all files to make them unique
# logging.info('---Start renaming all files---\n')
# unique_files(main_folder)
# logging.info('---Finished renaming all files---\n\n')

# run the function to copy all files to a single directory
# logging.info('---Start copying all files---\n')
# group_data(main_folder)
# logging.info('---Finished copying all files---\n\n')


# do train and validation split on all images
logging.info('---Start creating training and validation folders---\n')
train_val_split(main_folder)
logging.info('---Finished train-val split---\n\n')


# perform extra image augmentations with albumentation library
if args.augmentation == 'True':
    logging.info('---Start generating image augmentations---\n')
    img_augmentation(inner_path)
    logging.info('---Finished generating image augmentations---\n\n')


# create COCO JSON files for each train and val set
train_dir = os.path.join(inner_path, 'train')
val_dir = os.path.join(inner_path, 'val')

logging.info('---Start creating COCO JSON annotations---\n')

# if we are doing augmentation, use this list to add aug images to the json file
if args.augmentation == 'True':
    type_list = ['normal', 'augmented']
else:
    type_list = ['normal']

coco_pipeline(train_dir, category_ids, category_colors, multipolygon_ids, obj, type_list)
logging.info('Completed for training set!')
coco_pipeline(val_dir, category_ids, category_colors, multipolygon_ids, obj, type_list)
logging.info('Completed for validation set!\n\n')


# add in real images to our train and val folders, as well as annotation json if we are using them
if args.real_dir != 'False':
    real_images(object_id, args.real_dir, inner_path)

# create our test dataset suitable for input into Detectron2 database
# only for images for now
test_folder = os.path.join(str(os.getcwd()), args.test_dir)
test_folder = os.path.join(test_folder, 'images')
test_coco(test_folder, category_ids, obj)
logging.info('Completed for test set!\n\n')