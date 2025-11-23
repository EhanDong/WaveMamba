import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def list_all_files(startpath):  
    all_files = []  
    
    for root, dirs, files in os.walk(startpath):  
        for name in files:  
            if name[-4:]=='.jpg':
                all_files.append(name)  
    return all_files


def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        print("Model weights not found!")
        exit()
    model = YOLO(weights_path).to(device)
    model.fuse()
    model.info(verbose=False)
    return model

activation = {} 
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def process_images(path, model):
    # if not os.path.exists(path):
    #     print(f"Path {path} does not exist!")
    #     exit()
    #images_path=path+'images/val/'
    #image_path=path+'image/val/'
    #all_file=list_all_files(images_path)
    #all_file=['06144.jpg']
    #for i in tqdm(range(len(all_file))):
        #files=all_file[i]
        #print("1")
        pathrgb_ir=["/home/data1/FLIR/images/val/FLIR_08936_PreviewData.jpg","/home/data1/FLIR/image/val/FLIR_08936_PreviewData.jpg"]
        imgs=[]
        for img_file in pathrgb_ir:
            '''if not img_file.endswith(".jpg"):
                continue'''
            # img_path = os.path.join(path, img_file)
            img = cv2.imread(img_file)
            img = letterbox(img)[0]
            if img is None:
                print(f"Failed to load image {img_file}")
                continue
            imgs.append(img)
        maskrgb = imgs[0].copy()
        maskir = imgs[1].copy()
        imgs= np.concatenate((imgs[0], imgs[1]), axis=2)
        #model.eval()
        #print(model)
        #model.model[41].register_forward_hook(get_activation('cv2')) 
        model.predict(imgs,save=True,imgsz=640,visualize=True,obb=True,project="/home/YOLOv8")
        #bn3 = activation['cv2'] 
        #print(bn3.shape)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if os.path.exists('./detect') !=True:
        os.makedirs('./detect') 

    model = load_model("/home/flir/train4/weights/best.pt", device)
    process_images("/data1/datasets/visdrone/", model)

if __name__ == "__main__":
    main()