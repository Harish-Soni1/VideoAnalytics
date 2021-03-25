from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse
from detectron2.utils.visualizer import ColorMode
import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from flask import Response
from detectron2.utils.video_visualizer import VideoVisualizer
import requests
import numpy as np
from DesktopApplication.TrafficDetection import Ui_MainWindow



app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

def get_model(model_path, config_path, threshold):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_path

    return DefaultPredictor(cfg), cfg

def score_image(predictor: DefaultPredictor, image_url: str):
    image_reponse = requests.get(image_url)
    image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)

    # make prediction
    return predictor(image)

def prepare_pridctor():
    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    config_file = "modelsfiles/config.yml"
    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "modelsfiles/model_final.pth"
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

    #classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    #classes = MetadataCatalog.get("customtrain").thing_classes

    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    return (predictor)

predictor = prepare_pridctor()

@app.route("/api/score-image", methods=["POST"])
def process_score_image_request():
    image_url = request.json["imageUrl"]
    print(image_url)
    scoring_result = score_image(predictor, image_url)

    instances = scoring_result["instances"]
    scores = instances.get_fields()["scores"].tolist()
    #pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

    response = {
        "scores": scores,
        #"pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
        #"classes": classes
    }

    return jsonify(response)


instancemode = ColorMode.IMAGE
# @app.route("/video_prediction", methods=['GET'])
# @cross_origin()
def prediction_on_video(video):
    model = "modelsfiles/model_final.pth"
    config = "modelsfiles/config.yml"
    threshold = 0.5
    save_path = "output"
    predictor, cfg = get_model(model, config, threshold)
    parser = argparse.ArgumentParser(description='Detect objects from webcam images')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='',
                        help='Path to save the output. If None output won\'t be saved')
    args = parser.parse_args()
    print("Started")
    video_file = video #"/home/oem/Downloads/video.mp4"
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video stream or file")
    MetadataCatalog.get("customtrain").thing_classes = ['ear plugs',
                                                        'welding shield']
    metadata = MetadataCatalog.get("customtrain")

    while cap.isOpened():
        ret, image = cap.read()

        outputs = predictor(image)


        #v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        #VideoVisualizer
        #v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)

        video_visualizer = VideoVisualizer(metadata,instancemode)
        v = video_visualizer.draw_instance_predictions(image,outputs["instances"].to("cpu"))
        #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        if args.show:
            ui_main_window = Ui_MainWindow()
            ui_main_window.displayImage(cv2.imshow('object_detection', v.get_image()[:, :, ::-1]))
            #cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    #return Response("Successfully video")

def get_model_new():
    cfg = get_cfg()
    cfg.merge_from_file("modelsfiles/config.yml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "modelsfiles/model_final.pth"
    cfg.MODEL.DEVICE = 'cpu'

    return DefaultPredictor(cfg), cfg


@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.json["imageUrl"]
    image = cv2.imdecode(np.frombuffer(requests.get(image_url).content, np.uint8), cv2.IMREAD_COLOR)

    pred = predictor(image)

    instances = pred["instances"]
    scores = instances.get_fields()["scores"].tolist()
    pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_class = []
    # _, cfg = get_model()
    # for data in instances.pred_classes:
    #     num = data.item()
    #     print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
    }

    return response
















if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, port=port, host="127.0.0.1")

    #predictor, cfg = get_model_new()
