import torch, torchvision
import argparse
import logging
import time

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from src.preprocessing import *

print(torch.cuda.is_available())
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
inner_folder = os.listdir(main_folder)[0]
inner_path = os.path.join(main_folder, inner_folder)

test_folder = os.path.join(str(os.getcwd()), args.test_dir)
test_path = os.path.join(test_folder, 'videos')

# make use of this function to get number of classes from JSON file
category_ids, category_colors, count, object_of_interest = extract_json(main_folder)

# set cofigurations
configs = {
    'classes': count
            }


# settle the model configs
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
output_dir = os.path.join(inner_path, 'output')
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = configs['classes']
predictor = DefaultPredictor(cfg)

# run inference on videos
videos = os.listdir(test_path)
# get metadata from previous image runs
test_metadata = MetadataCatalog.get('test')

# read and write videos 1 by 1
# only can be run on MP4 files, not MOV files
for vid in videos:  
    start = time.time() 
    cap = cv2.VideoCapture(os.path.join(test_path,vid))

    # define codec and get video details
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_name = vid.replace('.mp4', '-result.mp4')
    output_path = os.path.join(test_path, output_name)

    # create video writer object for output of video file
    writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc,
                            fps=float(frames_per_second), 
                            frameSize=(width, height),
                            isColor=True,)

    print(f'Original FPS: {frames_per_second}')

    # show if video could be opened or not
    if (cap.isOpened() == False):
        logging.info(f'Error opening video file {vid}!')
    else:
        logging.info(f'Successfully opened video file {vid}.')

    v = VideoVisualizer(metadata=test_metadata)
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_1 = time.time()
            outputs = predictor(frame)
            stop_1 = time.time()
            diff_1 = float(stop_1 - start_1)
            # print(f'Time taken for predict step: {diff_1}')

            start_2 = time.time()
            out = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
            vis_frame = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
            stop_2 = time.time()
            diff_2 = float(stop_2 - start_2)
            # print(f'Time taken for draw step: {diff_2}')
            writer.write(vis_frame)
        else:
            break
    
    later = time.time()
    diff = int(later - start)
    fps = num_frames / diff
    logging.info(f'Inference FPS for {vid} is {fps} frames/second.')
    logging.info(f'Completed inference on {vid}.\n')
