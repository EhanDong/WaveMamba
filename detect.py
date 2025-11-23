
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from tqdm import tqdm
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]
    print(shape)  # current shape [height, width]
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

def process_images(path, model):
    # if not os.path.exists(path):
    #     print(f"Path {path} does not exist!")
    #     exit()
    images_path=path+'images/val/'
    image_path=path+'image/val/'
    all_file=list_all_files(images_path)
    #all_file=['00000.png']
    for i in tqdm(range(len(all_file))):
        files=all_file[i]
        #print("1")
        pathrgb_ir=[images_path+files,image_path+files]
        imgs=[]
        for img_file in pathrgb_ir:
            if not img_file.endswith(".jpg"):
                continue
            # img_path = os.path.join(path, img_file)
            img = cv2.imread(img_file)
            #print(img.shape)
            img = letterbox(img)[0]
            #print(img.shape)
            if img is None:
                print(f"Failed to load image {img_file}")
                continue
            imgs.append(img)
        maskrgb = imgs[0].copy()
        maskir = imgs[1].copy()
        imgs= np.concatenate((imgs[0], imgs[1]), axis=2)

        colors = [  
            [165, 0, 255],       
            [0, 255, 0],        
            [102, 255, 255],       
            [255, 165, 0],      
            [255, 255, 0],
            [255, 255, 166],
        ]
        #print("1")
        result = model(imgs,save=True,imgsz=640,visualize=False,obb=False,conf=0.5,project="/home/YOLOv8",iou=0.5)
        # cls, xywh = result[0].obb.cls, result[0].obb.xywh
        
        cls, xywh = result[0].boxes.cls, result[0].boxes.xywh
        class_conf=result[0].boxes.conf
        class_conf_=class_conf.detach().cpu().numpy()
        cls_, xywh_ = cls.detach().cpu().numpy(), xywh.detach().cpu().numpy()
        x=[]
        for pos, cls_value,cls_conf in zip(xywh_, cls_,class_conf_):
            pt1, pt2 = (np.int_([pos[0] - pos[2] / 2, pos[1] - pos[3] / 2]),
                        np.int_([pos[0] + pos[2] / 2, pos[1] + pos[3] / 2])) 
            color = colors[int(cls_value)]  
            #color = [0, 0, 255] if cls_value == 0 else [0, 255, 0]
            xfill=20
            yfill=15
            text_x=pt1[0]
            text_y=pt1[1]
            x1=4       
            if(text_x not in x and text_x==510):
                x1=12
            
            
            cv2.rectangle(maskrgb, tuple(pt1), tuple(pt2), color, x1)
            cv2.rectangle(maskir, tuple(pt1), tuple(pt2), color, x1)

            if(text_x not in x and text_x==510):
                x1=4

                     

            if(text_x+xfill>img.shape[1]):
                #print(text_x)
                text_x=img.shape[1]-30
            if(text_y-yfill<0):
                text_y=pt2[1]+10
            else :
                text_y-=2
            class_names = ["car","truck","bus","van","freight_car","motorcycle"]    
            class_name = class_names[int(cls_value)] if int(cls_value) < len(class_names) else "unknown"  
             
            font = cv2.FONT_HERSHEY_SIMPLEX  
            # font_scale = 0.6    
            font_color = color    
            # thickness = 2    
            font_scale = 0.8    
            # font_color = colors[int(label)]   
            thickness = 1    

            if(text_x in x and text_x==510):
                text_y+=40
                text_x-=90
            x.append(text_x)  
            
            
            text_y-=3
            text_x = max(text_x, 0) 
            text_y = max(text_y, 0)  
            
           
            
            # cv2.putText(maskrgb, class_name+f' {float(cls_conf):.2f}', (text_x, text_y-3), font, font_scale, font_color, thickness) 
            cv2.putText(maskrgb, class_name, (text_x, text_y), font, font_scale, font_color, thickness)  
            #cv2.putText(maskir, class_name+f' {float(cls_conf):.2f}', (text_x, text_y-3), font, font_scale, font_color, thickness)
            cv2.putText(maskir, class_name, (text_x, text_y), font, font_scale, font_color, thickness)  
            cv2.imwrite("/home/YOLOv8/keshihuadronelaji6/rgb"+files,maskrgb)
            cv2.imwrite("/home/YOLOv8/keshihuadronelaji6/ir"+files,maskir)
        print(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if os.path.exists('./detect') !=True:
        os.makedirs('./detect') 

    model = load_model("/home/YOLOv8/flir/train/weights/best.pt", device)
    process_images("/home/data1/FLIR/", model)

if __name__ == "__main__":
    main()

