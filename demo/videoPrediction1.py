from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse

def get_model(model_path, config_path, threshold):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path

    return DefaultPredictor(cfg), cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects from webcam images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='', help= 'Path to save the output. If None output won\'t be saved')
    args = parser.parse_args()

    predictor, cfg = get_model(args.model, args.config, args.threshold)

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    if args.save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))

    while cap.isOpened():
        ret, image = cap.read()

        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        if args.show:
            cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if args.save_path:
            out.write(image)
    cap.release()
    if args.save_path:
        out.release()
    cv2.destroyAllWindows()

################################################################################

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

from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer

import tqdm
import detectron2
from detectron2.utils.video_visualizer import VideoVisualizer
import time

# Load an image https://baystateelevator.us/wp-content/uploads/2018/05/451.jpg
# res = requests.get("https://i.pinimg.com/originals/f9/eb/1b/f9eb1b5d079fc6b3ba950c0a4e359535.jpg")
# image = np.asarray(bytearray(res.content), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)


config_file = "config_5000.yml"  # COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

cfg = get_cfg()
cfg.set_new_allowed(True)
# cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
cfg.merge_from_file(config_file)  # model_zoo.get_config_file(config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.91  # Threshold
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = "model_final.pth"  # model_zoo.get_checkpoint_url(config_file)
cfg.MODEL.DEVICE = "cuda"  # cpu or cuda

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
# output = predictor(image)
# print(output)
MetadataCatalog.get("customtrain").set(
    thing_colors=[(255, 0, 0), (0, 0, 0), (252, 22, 5), (252, 116, 5), (33, 245, 0), (252, 132, 3)])
MetadataCatalog.get("customtrain").thing_classes = ['ear plugs',
                                                    'hand gloves',
                                                    'helmet',
                                                    'mask',
                                                    'person',
                                                    'reflective jacket',
                                                    'safety goggles',
                                                    'safety harness belt',
                                                    'safety shoes',
                                                    'welding apron',
                                                    'welding shield']
metadata = MetadataCatalog.get("customtrain")

instancemode = ColorMode.IMAGE


# v = Visualizer(image[:, :, ::-1],
#                scale=4,
#                metadata=MetadataCatalog.get("customtrain"),
#                instance_mode=ColorMode.SEGMENTATION
#                )

# ins = output["instances"].to("cpu")
# ins.remove('pred_masks')
# v = v.draw_instance_predictions(ins)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("output", 500, 500)
# imgshow = cv2.imshow('output', v.get_image()[:, :, ::-1])

# cv2.waitKey(0)


#############################################################################################################################

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def run_on_video(video):
    """
    Visualizes predictions on frames of the input video.

    Args:
        video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
            either a webcam or a video file.

    Yields:
        ndarray: BGR visualizations of each video frame.
    """
    video_visualizer = VideoVisualizer(metadata, instancemode)

    def process_predictions(frame, predictions):
        # frame = cv2.flip(frame, 1)  # just for flipping the camera...
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if "panoptic_seg" in predictions:
        #     panoptic_seg, segments_info = predictions["panoptic_seg"]
        #     vis_frame = video_visualizer.draw_panoptic_seg_predictions(
        #         frame, panoptic_seg.to("cpu"), segments_info
        #     )
        # elif "instances" in predictions:
        predictions = predictions["instances"].to("cpu")
        # predictions.remove('pred_masks')
        vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
        # elif "sem_seg" in predictions:
        #     vis_frame = video_visualizer.draw_sem_seg(
        #         frame, predictions["sem_seg"].argmax(dim=0).to("cpu")
        #     )

        # Converts Matplotlib RGB format to OpenCV BGR format
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

        return vis_frame

    frame_gen = _frame_from_video(video)

    # if self.parallel:
    #     buffer_size = self.predictor.default_buffer_size
    #
    #     frame_data = deque()
    #
    #     for cnt, frame in enumerate(frame_gen):
    #         frame_data.append(frame)
    #         self.predictor.put(frame)
    #
    #         if cnt >= buffer_size:
    #             frame = frame_data.popleft()
    #             predictions = self.predictor.get()
    #             yield process_predictions(frame, predictions)
    #
    #     while len(frame_data):
    #         frame = frame_data.popleft()
    #         predictions = self.predictor.get()
    #         yield process_predictions(frame, predictions)
    # else:
    for frame in frame_gen:
        yield process_predictions(frame, predictor(frame))


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    for vis in tqdm.tqdm(run_on_video(cam)):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow("output", vis)
        if cv2.waitKey(1) == 27 or ord('q'):
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()