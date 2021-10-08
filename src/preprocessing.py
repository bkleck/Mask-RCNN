import glob
import os
import json
import re
from distutils.dir_util import copy_tree
from random import shuffle
import shutil
import logging
import albumentations as A
import cv2

from src.create_annotations import *


# function to extract label information from Unity JSON file
def extract_json(main_folder):
    # we can take any JSON file, so we take the first one
    folder = os.listdir(main_folder)[0]
    folder = os.path.join(main_folder, folder)

    # get the name of the Dataset folder
    dataset_folder = str([i for i in os.listdir(folder) if i.startswith('Dataset')][0])
    dataset_path = os.path.join(folder, dataset_folder)

    # read data from the json file
    with open(f'{dataset_path}/annotation_definitions.json') as f:
        d = json.load(f)

    # add the details into these dictionaries
    category_ids = {}
    category_colors = {}
    count = 0

    # read the Unity JSON and save to the above dictionaries
    for object in d['annotation_definitions'][0]['spec']:

        # add name of object as key, and id number as value
        category_ids[object['label_name']] = count

        # get pixel values and multiply them by 255
        red = int(object['pixel_value']['r'] * 255)
        green = int(object['pixel_value']['g'] * 255)
        blue = int(object['pixel_value']['b'] * 255)
        tup = str((red, green, blue))

        # add tuple of pixel values as key, and id number as value
        category_colors[tup] = count

        # increase the id value after each object
        count += 1
        
    return category_ids, category_colors




# function to rename files to make every file unique
# output image files from Unity are not unique
# rename every file with their folder prefix
def unique_files(main_folder):
    # run through every folder generated by the different scenes in Unity
    for folder in os.listdir(main_folder):
        path = os.path.join(main_folder, folder)
        inner_folders = os.listdir(path)

        # get the name of the RGB and Semantic folders
        img_folder = str([i for i in inner_folders if i.startswith('RGB')][0])
        semantic_folder = str([i for i in inner_folders if i.startswith('Semantic')][0])

        # rename each file to include the folder name in its name
        img_path = os.path.join(path, img_folder)
        img_files = os.listdir(img_path)
        for file in img_files:
            new_file = f'{folder}_{file}'
            os.rename(os.path.join(img_path, file), os.path.join(img_path, new_file))

        semantic_path = os.path.join(path, semantic_folder)
        semantic_files = os.listdir(semantic_path)
        for file in semantic_files:
            new_file = f'{folder}_{file}'
            os.rename(os.path.join(semantic_path, file), os.path.join(semantic_path, new_file))
        
        logging.info(f'Completed renaming of files for {folder} folder!')



# function to add all files into a single directory
# this helps with our train-validation split later on
def group_data(main_folder):
    # create a new final folder with sub-folders of 'images' and 'semantic'
    main_path = os.path.join(main_folder, 'final')
    final_path = os.path.join(main_path, 'images')
    final_path_2 = os.path.join(main_path, 'semantic')
    try:
        os.mkdir(main_path)
        os.mkdir(final_path)
        os.mkdir(final_path_2)
    except FileExistsError:
        # directory already exists
        pass

    for folder in os.listdir(main_folder):
        if folder != 'final': # no need to do for this folder
            path = os.path.join(main_folder, folder)
            inner_folders = os.listdir(path)

            # get the name of the RGB and Semantic folders
            img_folder = str([i for i in inner_folders if i.startswith('RGB')][0])
            img_path = os.path.join(path, img_folder)

            semantic_folder = str([i for i in inner_folders if i.startswith('Semantic')][0])
            semantic_path = os.path.join(path, semantic_folder)

            # copy all items from these paths to the final folders
            copy_tree(img_path, final_path)
            copy_tree(semantic_path, final_path_2)

            logging.info(f'Completed copying files from {folder} folder!')



# split into training and validation folders
def train_val_split(main_folder):
    main_path = os.path.join(main_folder, 'final')
    img_path = os.path.join(main_path, 'images')
    semantic_path = os.path.join(main_path, 'semantic')

    # # create required folders (train & val)
    train_path = os.path.join(main_path, 'train')
    train_img_path = os.path.join(train_path, 'images')
    train_seg_path = os.path.join(train_path, 'segmentation')

    val_path = os.path.join(main_path, 'val')
    val_img_path = os.path.join(val_path, 'images')
    val_seg_path = os.path.join(val_path, 'segmentation')

    try:
        os.mkdir(train_path)
        os.mkdir(train_img_path)
        os.mkdir(train_seg_path)

        os.mkdir(val_path)
        os.mkdir(val_img_path)
        os.mkdir(val_seg_path)

    except FileExistsError:
        # directory already exists
        pass


    # shuffle images in folder randomly
    images = os.listdir(img_path)
    shuffle(images)

    # split into train and val sets
    train_ratio = 0.8
    no_of_images = int(train_ratio * len(images))
    train_set = images[0: no_of_images]
    val_set = images[no_of_images: -1]

    # copy images and segmentations into new train and val folders
    for name in train_set:
        file = os.path.join(img_path, name)
        shutil.copy(file, os.path.join(train_img_path, name))

        seg_name = name.replace('rgb', 'segmentation')
        seg_file = os.path.join(semantic_path, seg_name)
        shutil.copy(seg_file, os.path.join(train_seg_path, seg_name))

    logging.info(f'Completed copying image and segmentation files to train folder!')

    for name in val_set:
        file = os.path.join(img_path, name)
        shutil.copy(file, os.path.join(val_img_path, name))

        seg_name = name.replace('rgb', 'segmentation')
        seg_file = os.path.join(semantic_path, seg_name)
        shutil.copy(seg_file, os.path.join(val_seg_path, seg_name))

    logging.info(f'Completed copying image and segmentation files to validation folder!')



# get 'images' and 'annotations' info from the RGB & segmentation images
# we will extract image dimensions and polygons from the segmentation images
# and include the filename using RGB images
def images_annotations_info(main_directory, category_ids, category_colors, multipolygon_ids, obj):
    mask_path = os.path.join(main_directory, 'segmentation')
    img_path = os.path.join(main_directory, 'images')

    # This id will be automatically increased as we go
    annotation_id = 0
    # ask for user input on what object this is and get its index
    image_id = category_ids[obj]
    annotations = []
    images = []

    for mask_image in os.listdir(mask_path):
        if mask_image.endswith(".png"):
            short_mask_name = mask_image
            mask_image = os.path.join(mask_path, mask_image)

            # The mask image is in Semantic folder but the original image is in RGB folder.
            # We make a reference to the original file 
            # compare RGB vs segmentation corresponding images using their names
            rgb_name = short_mask_name.replace('segmentation', 'rgb') # COCO format does not have path in name
            original_file_name = os.path.join(img_path, rgb_name)

            # we need to open both original and augmented image
            list1 = ['normal', 'augmented']

            for i in list1:
                if i == 'normal':
                    rgb_name = rgb_name
                else:
                    rgb_name = rgb_name.replace('.png', '_aug.png')

                # Open the image and (to be sure) we convert it to RGB
                mask_image_open = Image.open(mask_image).convert("RGB")
                w, h = mask_image_open.size

                # "images" info 
                image = create_image_annotation(rgb_name, w, h, image_id)
                images.append(image)

                sub_masks = create_sub_masks(mask_image_open, w, h)
                for color, sub_mask in sub_masks.items():
                    category_id = category_colors[color]

                    # "annotations" info
                    polygons, segmentations = create_sub_mask_annotation(sub_mask)

                    # Check if we have classes that are a multipolygon
                    if category_id in multipolygon_ids:
                        # Combine the polygons to calculate the bounding box and area
                        multi_poly = MultiPolygon(polygons)
                                        
                        annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                        annotations.append(annotation)
                        annotation_id += 1
                    else:
                        for i in range(len(polygons)):
                            # Cleaner to recalculate this variable
                            segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                            
                            annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                            
                            annotations.append(annotation)
                            annotation_id += 1
                image_id += 1
    return images, annotations, annotation_id



# this pipeline runs the function above to extract image details
# then it converts it to COCO format and outputs a json file
def coco_pipeline(main_directory, category_ids, category_colors, multipolygon_ids, obj):
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(main_directory, category_ids, category_colors, multipolygon_ids, obj)

    # put json into same folder as RGB images
    img_path = os.path.join(main_directory, 'images')
    annot_path = img_path
    mask_path = os.path.join(main_directory, 'segmentation')

    with open(f"{annot_path}/annotations.json", "w") as outfile:
        json.dump(coco_format, outfile)

    logging.info("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))



# Image Augmentation
# note that OpenCV reads images in BGR, matplotlib displays images in RGB
# this function adds a series of augmentations to our original images
# outputs new images with extra suffix "_aug"
def img_augmentation(main_folder):
    train_path = os.path.join(main_folder, 'final/train/images')
    val_path = os.path.join(main_folder, 'final/val/images')

    # start with the training set
    train_imgs = os.listdir(train_path)

    # read each image path with cv2 to convert to image array
    train_list = [cv2.imread(os.path.join(train_path, i)) for i in train_imgs]
    # convert from BGR to RGB
    # train_list = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in train_list]

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.HueSaturationValue(p=0.3),
        A.RandomContrast(p=0.3, limit=0.5),
        A.RandomBrightness(p=0.3, limit=0.4),
        A.ToGray(p=0.5),
        A.GaussNoise(var_limit=(50,70), p=0.3),
        A.ISONoise(p=0.3, intensity=(0.3,0.7)),
        A.MotionBlur(p=0.3),
        A.GaussianBlur(p=0.3)
    ])

    # run augmentation on every image and save into same folder
    # convert back from RGB to BGR
    # save it with new name (add '_aug' behind every file name)
    for idx,image in enumerate(train_list):
        transformed = transform(image=image)['image']
        # transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        new_name = train_imgs[idx].replace('.png', '') + '_aug.png'
        cv2.imwrite(os.path.join(train_path, new_name), transformed)
    logging.info(f'Completed augmentations for train folder!')


    # do for validation set
    val_imgs = os.listdir(val_path)
    val_list = [cv2.imread(os.path.join(val_path, i)) for i in val_imgs]
    # val_list = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in val_list]

    for idx,image in enumerate(val_list):
        transformed = transform(image=image)['image']
        # transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        new_name = val_imgs[idx].replace('.png', '') + '_aug.png'
        cv2.imwrite(os.path.join(val_path, new_name), transformed)  
    logging.info(f'Completed augmentations for val folder!')



# functions below are for the test dataset, which are in a separate folder


# function to get "images" and "annotations" info from test dataset
def test_json(main_directory, category_ids, obj):
    img_path = main_directory

    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = category_ids[obj]
    annotations = []
    images = []

    for image in os.listdir(img_path):
        if image.endswith(".jpg"):
            short_name = image
            image = os.path.join(img_path, image)

            # Open the image and (to be sure) we convert it to RGB
            image_open = Image.open(image).convert("RGB")
            w, h = image_open.size

            # "images" info 
            image_info = create_image_annotation(short_name, w, h, image_id)
            images.append(image_info)

            image_id += 1
    return images, annotations, annotation_id


# function to create COCO JSON for test dataset
def test_coco(main_directory, category_ids, obj):
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = test_json(main_directory, category_ids, obj)

    # put json into same folder as RGB images
    annot_path = main_directory

    with open(f"{annot_path}/annotations.json", "w") as outfile:
        json.dump(coco_format, outfile)

    logging.info("Created %d annotations for images in folder" % (annotation_cnt))