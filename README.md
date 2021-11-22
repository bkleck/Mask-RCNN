# Mask-RCNN
*Instance Segmentation for Object Detection*
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/142635960-72766b81-8dc1-46c7-83b3-86314e7a292b.jpg' width='500' height='300'>

## Table of Contents
* [Introduction](#introduction)
* [Documentation](#documentation)
  * [Data Processing](#1-data-processing)
  * [Image Augmentation](#2-image-augmentation)
  * [Model Training](#3-model-training)
  * [Model Inference](#4-model-inference)


## Introduction
This project implements the **_Mask RCNN model_** using Facebook's [Detectron2](https://github.com/facebookresearch/detectron2) library. The purpose was to **_create segmentation masks_** on frames of images or videos, in order to predict the object class, as well as utilize the orientation of the mask for **_robotic pick-and-place operations_**. 

To reduce the laborious process of labelling data for model training, I also connected it to a **_synthetic data pipeline_**, built with Unity and C#. This pipeline will be described in my other [repository](https://github.com/bkleck/SyntheticData). Synthetic images produced were uploaded via the front-end iOS application to an AWS S3 bucket, which was then used to trigger the Python backend on EC2 for data processing and model development.
<br/> 
<br/> 

## Documentation
### 1) Data Processing
- Python Scripts: *augmentation_pipeline.py, preprocessing.py, create_annotations.py*

The output from the [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) API is not in the appropriate format for the Mask-RCNN model, hence I will write custom scripts to convert our data output into the **_COCO XML format_**. This will also allow our synthetic data to be used with other models, if there is a need for it.

Firstly, I will extract the **_high level information_** from the dataset of interest. For each class of object, I will extract the corresponding **_ID number, name, and colour_** (using pixel value). This will be used later on for identification of the object and appropriate classification by the model.

Next, I will need to **_create training & validation datasets_** for my model. To reduce bias in my model, I will **_shuffle the RGB images randomly_** and split them into training and validation folders with the **_train-val ratio of 0.8 : 0.2_**. As the segmentation images are matched to their respective RGB images by their unique ID number, we will also split the segmentation images into training and validation folders using this ID.

<p align="center">
 <img src='https://user-images.githubusercontent.com/77097236/142641690-97a2ea93-4d48-46d4-a787-9dbcc72f4894.png' width='250' height='150'>
 <img src='https://user-images.githubusercontent.com/77097236/142641857-4382bf76-2bc2-43d8-9806-1c48a9278c7e.png' width='250' height='150'>
 <br/> 
 <i>Example of a RGB image with its corresponding segmentation image</i>
</p>

<br/> 

Lastly, we will extract the **_low level information_** from each image that we have, mainly using the [OpenCV](https://github.com/opencv/opencv) library. From the RGB image, I will extract the **_file name, width, height and class ID_**, and they will be formatted into a dictionary and become one entry within the JSON file. From the segmentation image, I will make use of the [shapely](https://shapely.readthedocs.io/en/stable/manual.html) library to extract the **_polygon points_** from the image into the JSON file, and this will be used later on to construct the mask.
<br/> 
<br/> 

### 2) Image Augmentation
- Python Scripts: *augmentation_pipeline.py, preprocessing.py*
After looking at the model results, I realise that the model **_could not perform well in poor environmental conditions_**, such as low-lighting or blur caused by the camera lens, hence I decided to add in image augmentation to my synthetic dataset in order to **_increase variance and flexibility_** in different environments. After researching on various libraries, I went with [Albumentations](https://github.com/albumentations-team/albumentations) because of its **_faster speed, huge variety of augmentations and ease of use_**.

 <img src='https://user-images.githubusercontent.com/77097236/142648920-f4b6e476-69c0-4eeb-b0b3-6d9af629e15e.png' width='300' height='200'>
 
I created an **_augmentation copy of each image_**, hence my dataset doubled in size after this step. The transformations I utilized were HueSaturation, Contrast, Brightness, GrayScale, GaussianNoise, ISONoise, MotionBlur and GaussianBlur. I applied a probability to each of this transformations, hence there is a likelihood of more than one transformation being applied to each image, hence increasing noise to improve our model variance. Some examples are shown below:

<p align="center">
  <img src='https://user-images.githubusercontent.com/77097236/142650630-1955b9c5-73f6-4309-95c9-2d3978906ac4.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142650725-ccfcd3fa-9f98-4d59-8efb-c296c3d80b61.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142650797-80c4518b-fa89-4cf3-be44-b81aad5905ce.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142650848-2930d14a-9804-46ce-bab3-4ed2d1e7192c.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142650873-9ffef3f0-7f62-41fc-b60a-af8a077d891e.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142650913-862d46f2-94b2-4f6f-8cb1-3b53610d779d.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142651839-e14a871b-ad54-4fa0-a420-352a751ed5f6.png' width='200' height='125'>
  <img src='https://user-images.githubusercontent.com/77097236/142652135-91e92c65-d7a7-4e0b-bc08-f5ada0b8ac3b.png' width='200' height='125'>
</p>

<br/> 

### 3) Model Training
- Python Scripts: *mask_rcnn_train.py*
As we will be utilizing the MASK-RCNN model for our instance segmentation task, I will make use of **_Facebook's Detectron2 library_** for their seamless integration of models into the entire training and inference pipeline. The exact model config I utilized is [here](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml). 

Firstly, I will need to **_register my training and validation datasets_** to Detectron's catalog to fit their workflow. As I have previously converted my datasets into the **_standard COCO format_**, this can be done easily with the following code:
```
register_coco_instances("train", {}, f'{train_path}/annotations.json', train_path)
dataset_dicts = DatasetCatalog.get('train')
train_metadata = MetadataCatalog.get('train')
```
I will also extract the number of classes from our JSON file. Although the model has a multi-class head, we will only train it to **_identify 1 class_** to fit the client's workflow. To standardize the workflow and **_reduce the need for the end-user to tweak the hyper-parameters_**, I have done my own hyper-parameter tuning and settled with the following:
- 2 images per batch
- 0.00025 learning rate
- 1000 epochs
- 64 batch size per image

For each object, I will be creating an output folder to store the **_checkpoints and model weights_** after training. After completing the configurations for the model, we will start the training with:
```
trainer = DefaultTrainer(cfg) 
trainer.train()
```
I will keep any mask predictions with **_confidence above 80%_**, and make use of our trained model to perform validation on the val dataset. Results will be printed in the command line, with various metrics such as **_accuracy, AP, AR, IoU_**.

<br/> 

### 4) Model Inference
- Python Scripts: *mask_rcnn_inference.py, video_inference.py*
Now I will perform inference with the model I just trained. Configurations will be similar to that used in the training phase, except that we will make use of the test dataset instead, and load the model weights from the **_"model_final.pth"_** file in the output folder.

For each image in the test dataset, I will read the image with OpenCV and make use of Detectron's DefaultPredictor and Visualizer to **_output predictions and draw instance masks_** on the image respectively. This will also be done for each video we have using **_OpenCV's video modules_**, with conversion between GBR and RGB as OpenCV utilises the unconventional BGR format. 
