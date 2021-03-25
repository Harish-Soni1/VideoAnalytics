from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import requests
import cv2
import torch
import numpy as np
import requests
import multiprocessing as mp
from collections import deque


# import tqdm
###### import the visualization demo from predictor ########
import detectron2
from detectron2.utils.video_visualizer import VideoVisualizer
import time
# Load an image
res = requests.get("https://i.pinimg.com/originals/f9/eb/1b/f9eb1b5d079fc6b3ba950c0a4e359535.jpg")
image = np.asarray(bytearray(res.content), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)


config_file = "modelsfiles/config.yml" #
#config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()
cfg.set_new_allowed(True)
#cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.merge_from_file(config_file) # model_zoo.get_config_file(config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 # Threshold
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = "modelsfiles/model_final.pth" #
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.DEVICE = "cuda" # cpu or cuda


# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
output = predictor(image)
print(output)
v = Visualizer(image[:, :, ::-1],
               scale=1,
               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
               instance_mode=ColorMode.IMAGE
               )
ins = output["instances"].to("cpu")
#ins.remove('pred_masks')
v = v.draw_instance_predictions(ins)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
imgshow = cv2.imshow('output', v.get_image()[:, :, ::-1])

cv2.waitKey(0)
