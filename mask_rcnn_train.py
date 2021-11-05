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
args = parser.parse_args()

main_folder = os.path.join(str(os.getcwd()), args.input_dir)

# create training and validation paths
train_path = os.path.join(main_folder, 'final/train/images')
val_path = os.path.join(main_folder, 'final/val/images')



# register new datasets to the detectron catalog
register_coco_instances("train", {}, f'{train_path}/annotations.json', train_path)

dataset_dicts = DatasetCatalog.get('train')
train_metadata = MetadataCatalog.get('train')

register_coco_instances("val", {}, f'{val_path}/annotations.json', val_path)
val_dicts = DatasetCatalog.get('val')
val_metadata = MetadataCatalog.get('val')


# if you want to visualize sample data input
# for d in random.sample(dataset_dicts, 10):
#     file = d['file_name']
#     img = cv2.imread(file)
#     visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.3)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('Window', out.get_image()[:, :, ::-1])
#     cv2.waitKey()

# make use of this function to get number of classes from JSON file
category_ids, category_colors, count = extract_json(main_folder)

# Fine-tune a pretrained model

# set cofigurations
configs = {'num_workers': 2,
            'image_per_batch': 2,
            'lr': 0.00025,
            'epochs': 1000,
            'batch_size_per_img': 64,
            'classes': count
            }
            
logging.info('These are the model configurations:')
for k,v in configs.items():
    print(f'{k}: {v}')
print('')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ()
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = configs['num_workers']
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = configs['image_per_batch']
cfg.SOLVER.BASE_LR = configs['lr'] # pick a good LR
cfg.SOLVER.MAX_ITER = configs['epochs']    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configs['batch_size_per_img']   # faster, and good enough for this dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = configs['classes']  # only has one class (labo)

output_dir = os.path.join(main_folder, 'output')
cfg.OUTPUT_DIR = output_dir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# start trainer
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


logging.info('\nTraining is completed.\n')


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


# perform validation on val dataset
evaluator = COCOEvaluator("val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
logging.info('\nValidation is completed.\n')