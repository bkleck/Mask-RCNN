import torch, torchvision
import argparse
import logging

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
args = parser.parse_args()

main_folder = os.path.join(str(os.getcwd()), args.input_dir)


# set cofigurations
configs = {
    'classes': 2
            }


# settle the model configs
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
output_dir = os.path.join(main_folder, 'output')
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = configs['classes']
predictor = DefaultPredictor(cfg)


# get metadata from previous image runs
test_metadata = MetadataCatalog.get('test')

# run inference on webcam
cap = cv2.VideoCapture(0)

# define codec and get video details
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'width:{width}, height:{height}, fps:{frames_per_second}')

# show if video could be opened or not
if (cap.isOpened() == False):
    logging.info(f'Error opening webcam!')
else:
    logging.info(f'Successfully opened webcam!')

count = 0

while (cap.isOpened()):
    count += 1
    ret, frame = cap.read()

    # only run inference on every 5 frames
    if ret == True and count == 5:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = predictor(frame)
        v = VideoVisualizer(metadata=test_metadata)

        out = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        vis_frame = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
        cv2.imshow(vis_frame)
        # reset the counter
        count = 0

    elif ret == False:
        break
    
logging.info(f'Completed inference on webcam.')