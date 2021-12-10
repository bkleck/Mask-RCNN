# Mask-RCNN
*Instance Segmentation for Object Detection*
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/142635960-72766b81-8dc1-46c7-83b3-86314e7a292b.jpg' width='500' height='300'>

## Table of Contents
* [Introduction](#introduction)
* [How to Use](#how-to-use)
* [Documentation](#documentation)
  * [Data Processing](#1-data-processing)
  * [Image Augmentation](#2-image-augmentation)
  * [Model Training](#3-model-training)
  * [Model Inference](#4-model-inference)
* [Detailed Pipeline](#detailed-pipeline)


## Introduction
This project implements the **_Mask RCNN model_** using Facebook's [Detectron2](https://github.com/facebookresearch/detectron2) library. The purpose was to **_create segmentation masks_** on frames of images or videos, in order to predict the object class, as well as utilize the orientation of the mask for **_robotic pick-and-place operations_**. 

To reduce the laborious process of labelling data for model training, I also connected it to a **_synthetic data pipeline_**, built with Unity and C#. This pipeline will be described in my other [repository](https://github.com/bkleck/SyntheticData). Synthetic images produced were uploaded via the front-end iOS application to an AWS S3 bucket, which was then used to trigger the Python backend on EC2 for data processing and model development.
<br/> 
<br/> 

## How to Use
1) Environment has been setup in the **_CustomModels EC2 instance_**. If you want to set it up again, please install the dependencies with the requirements.txt. Start the EC2 instance, activate the **_pytorch_36 environment_** and go into the synthetic directory.
```
ssh -i "augmentusubuntu.pem" ubuntu@{EC2 instance}
cd synthetic
conda activate pytorch_p36
```
<br/> 

2) Upload synthetic image files from the **_Augmentus MainApp_** to the **_augmentus-synthetic S3 bucket_** by pressing the Upload to Cloud button.
<br/> 

3) Sync the bucket to this EC2 instance.
```
aws s3 sync s3://augmentus-synthetic .
```
<br/> 

4) Run the first python file to start the **_data processing, augmentation and add in real images_** to complement the synthetic data. 
Input parameters:
- input_dir = path to the synthetic data folder (e.g. "data/listerine#10-12-21#11 06 AM")
- test_dir = path to real images and videos for testing of model accuracy (e.g. data/listerine_test)
- real_dir = path to folder containing real images and annotations manually done using [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) (e.g. data/listerine_real), default="False"
- augmentation = whether you want to perform augmentation on the dataset ("True"/"False"), default="False"
```
python augmentation_pipeline.py --input_dir="data/listerine#18-11-21#10 45 AM" --test_dir=data/listerine_test
```
<br/> 

5) Run the second python file to start the **_training of the Mask-RCNN model_**.
```
python mask_rcnn_train.py --input_dir="data/listerine#18-11-21#10 45 AM"
```
<br/> 

6) Run the last 2 python files to run **_inference on the images and videos_** in the test directory respectively.
```
python mask_rcnn_inference.py --input_dir="data/listerine#18-11-21#10 45 AM" --test_dir=data/listerine_test
python video_inference.py --input_dir="data/listerine#18-11-21#10 45 AM" --test_dir=data/listerine_test
```
<br/> 

7) To display **_training statistics_**, go into the output directory and activate tensorboard.
```
tensorboard --logdir=./ --port=6006 --bind_all
```


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

<br/> 

## Detailed Pipeline
<p align="center">
 <img src='https://user-images.githubusercontent.com/77097236/145533449-12e14702-ecbf-4ff9-bc35-122a8dc59fe3.png'>
</p>

### 1) Synthetic Data Generation
In this part, I will upload an object into Unity and make use of the Perception library to generate randomized scenes to create a synthetic dataset. Click the 'Upload to Cloud' button to send images to the synthetic S3 bucket. 

**Input:**
- new Semantic Segmentation Label Config for each new project, attach it to the Main Camera in Synthetic Scene
- input object name (exactly the same as the LabelConfig) and OBJ file
<img src='https://user-images.githubusercontent.com/77097236/145534402-05f0b627-0306-44dd-98e1-dd1eba593b54.png' width='300' height='250'>

**Output:**
- RGB images in the RGB folder and Mask images in the SemanticSegmentation folder
- JSON files with various metrics in the Dataset folder

### 2) Image Processing
In this part, I will perform the data processing, augmentation and complement the synthetic dataset with manually annotated real images. Manual annotation was done through the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/), thus it will only work with the JSON files produced using this site.

**Input:**
- Synthetic data folder from S3 bucket containing output from Unity Perception
- Test data folder with images and videos in respective sub-folders
- Real data folder with images

**Output:**
- Train and validation folders containing RGB, augmented and real images in the images sub-folder, and mask images in the segmentation sub-folder
- JSON files in COCO XML format will also be stored in respective images sub-folder

### 3) Mask RCNN Training
In this part, we will train our model to perform instance segmentation. 

**Input:** 
- Train and validation folders with images and COCO JSON file

**Output:**
- Model weights for each object we trained (single-class training)
- Tensorboard logging of results

### 4) Mask RCNN Inference
In this part, we will evaluate our model performance by running inference on real image and video samples.

**Input:**
- Model weight
- Test dataset

**Output:**
- Image and video results with mask output on frames
