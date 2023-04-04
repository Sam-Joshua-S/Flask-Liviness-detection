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
import pandas as pd
warnings.filterwarnings("ignore")
model = create_model("tf_efficientnet_b3_ns")
data1 =[]
data2=[]

def liveness(data,camera=0):
    vid = cv2.VideoCapture(camera)
    model.eval()
    flag=[]
    
    t_end = 20
    while t_end:
        prediction=-1
        img_encoding1=[]
        ret, frame = vid.read()
        try:
            rgb_img1=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # siamese network
            img_encoding1=face_recognition.face_encodings(rgb_img1)[0]
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
            t_end=t_end-1
            data.append(prediction)
            prediction = np.argmax(prediction)

                
    return data

if __name__ == "__main__":
    df=pd.DataFrame(np.array(liveness(data1)))
    df.to_csv("real.csv")
    _ = input("Show me your picture (Enter to continue)")
    df=pd.DataFrame(np.array(liveness(data2)))
    df.to_csv("reel.csv")