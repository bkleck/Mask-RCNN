# Mask-RCNN
*Instance Segmentation for Object Detection*
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/142635960-72766b81-8dc1-46c7-83b3-86314e7a292b.jpg' width='500' height='300'>

## Table of Contents
* [Introduction](#introduction)
* [Documentation](#documentation)
  * [Data Processing](#1-data-processing)


## Introduction
This project implements the **_Mask RCNN model_** using Facebook's [Detectron2](https://github.com/facebookresearch/detectron2) library. The purpose was to **_create segmentation masks_** on frames of images or videos, in order to predict the object class, as well as utilize the orientation of the mask for **_robotic pick-and-place operations_**. 

To reduce the laborious process of labelling data for model training, I also connected it to a **_synthetic data pipeline_**, built with Unity and C#. This pipeline will be described in my other [repository](https://github.com/bkleck/SyntheticData). Synthetic images produced were uploaded via the front-end iOS application to an AWS S3 bucket, which was then used to trigger the Python backend on EC2 for data processing and model development.
<br/> 

## Documentation
### 1) Data Processing
- Python Scripts: *augmentation_pipeline.py, preprocessing.py, create_annotations.py*

The output from the [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) API is not in the appropriate format for the Mask-RCNN model, hence I will write custom scripts to convert our data output into the **_COCO XML format_**. This will also allow our synthetic data to be used with other models, if there is a need for it.

Firstly, I will extract the **_high level information_** from the dataset of interest. For each class of object, I will extract the corresponding **_ID number, name, and colour_** (using pixel value). This will be used later on for identification of the object and appropriate classification by the model.

Next, I will need to **_create training & validation datasets_** for my model. To reduce bias in my model, I will **_shuffle the RGB images randomly_** and split them into training and validation folders with the **_train-val ratio of 0.8 : 0.2_**. As the segmentation images are matched to their respective RGB images by their unique ID number, we will also split them segmentation images into training and validation folders using this ID.

<p float="left">
 <img src='https://user-images.githubusercontent.com/77097236/142641690-97a2ea93-4d48-46d4-a787-9dbcc72f4894.png' width='250' height='150'>
 <img src='https://user-images.githubusercontent.com/77097236/142641857-4382bf76-2bc2-43d8-9806-1c48a9278c7e.png' width='250' height='150'>
</p>

*Example of a RGB image with its corresponding segmentation image*

Lastly, we will extract the **_low level information_** from each image that we have, mainly using the [OpenCV](https://github.com/opencv/opencv) library. From the RGB image, I will extract the **_file name, width, height and class ID_**, and they will be formatted into a dictionary and become one entry within the JSON file. From the segmentation image, I will make use of the [shapely](https://shapely.readthedocs.io/en/stable/manual.html) library to extract the **_polygon points_** from the image into the JSON file, and this will be used later on to construct the mask.
