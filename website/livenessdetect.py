import cv2
import face_recognition
import albumentations as albu
import torch
from datasouls_antispoof.pre_trained_models import create_model
from albumentations.pytorch.transforms import ToTensorV2
import time
from datasouls_antispoof.class_mapping import class_mapping
import warnings
import numpy as np
import pickle
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'model', 'dif.pickle')
with open(data_path, 'rb') as f:
    dt_model =pickle.load(f)
warnings.filterwarnings("ignore")
model = create_model("tf_efficientnet_b3_ns")
model.eval()

def face_recogni(vid):
    t_end = time.time() + 20
    
    while time.time() < t_end:
        try:
            ret, frame = vid.read()
            rgb_img1=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # siamese network
            img_encoding1=face_recognition.face_encodings(rgb_img1)[0]
            return img_encoding1
        except IndexError as e:
            pass
    return []

def liveness(vid,cnt):
    model.eval()
    flag=[]
    t_end = time.time() + 20
    while time.time() < t_end and cnt!=0:
        prediction=-1
        img_encoding1=[]
        ret, frame = vid.read()
        try:
            rgb_img1=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # siamese network
            img_encoding1=face_recognition.face_encodings(rgb_img1)[0]
            print("Got Face")
            cnt=cnt-1
        except IndexError as e:
            pass
        if img_encoding1!=[]:
            image_replay =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                                    albu.CenterCrop(height=400, width=400), 
                                    albu.Normalize(p=1), 
                                    albu.pytorch.ToTensorV2(p=1)], p=1)
            with torch.no_grad():
                prediction = model(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]
            prediction = dt_model.predict(prediction.reshape(1,-1))
            if prediction==1:
                print("Liveness is detected")
                return True,img_encoding1
    return False,flag

if __name__ == "__main__":
    print(liveness(0))