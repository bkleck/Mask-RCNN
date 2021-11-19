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
