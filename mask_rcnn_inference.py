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
test_folder = os.path.join(str(os.getcwd()), args.test_dir)
test_path = os.path.join(test_folder, 'images')

# make use of this function to get number of classes from JSON file
category_ids, category_colors, count = extract_json(main_folder)

# set cofigurations
configs = {
    'classes': count
            }


# register test dataset to Detectron2
register_coco_instances("test", {}, f'{test_path}/annotations.json', test_path)
test_dicts = DatasetCatalog.get('test')
test_metadata = MetadataCatalog.get('test')

# settle the model configs
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
output_dir = os.path.join(main_folder, 'output')
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = configs['classes']
predictor = DefaultPredictor(cfg)

# run inference on test images and output them with corresponding labels
for d in test_dicts:    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata= test_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out.save('result_' + d['file_name'])
