import torch
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='road', help='Model type to train (road or object)')
args = parser.parse_args()

def train_road_model():

    DATA_PATH = 'cctv-roadseg-dataset/data.yaml'

    model = YOLO('yolo26l-seg.pt')

    model.train(data=DATA_PATH, epochs=50, imgsz=640, batch=-1, device=0)

def train_crosswalk_model():

    DATA_PATH = 'cctv-crosswalk-dataset/data.yaml'

    model = YOLO('yolo26l-seg.pt')

    model.train(data=DATA_PATH, epochs=50, imgsz=640, batch=-1, device=0, patience=20)

def train_object_model():

    DATA_PATH = 'cctv-object-dataset/data.yaml'

    model = YOLO('yolo26s-seg.pt')

    model.train(data=DATA_PATH, epochs=50, imgsz=640, batch=-1, device=0)


if __name__ == "__main__":
    if args.model == "road":
        train_road_model()
    elif args.model == "crosswalk":
        train_crosswalk_model()
    elif args.model == "object":
        train_object_model()
    