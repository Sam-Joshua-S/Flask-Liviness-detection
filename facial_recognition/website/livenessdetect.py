import cv2
import albumentations as albu
import torch
from datasouls_antispoof.pre_trained_models import create_model
from albumentations.pytorch.transforms import ToTensorV2
import time
from datasouls_antispoof.class_mapping import class_mapping
import warnings
import numpy as np
warnings.filterwarnings("ignore")
model = create_model("tf_efficientnet_b3_ns")
model.eval()
def liveness(camera=0):
    vid = cv2.VideoCapture(camera)
    model.eval()
    t_end = time.time() + 20
    while time.time() < t_end:
        ret, frame = vid.read()
        image_replay =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                                albu.CenterCrop(height=400, width=400), 
                                albu.Normalize(p=1), 
                                albu.pytorch.ToTensorV2(p=1)], p=1)
        with torch.no_grad():
            prediction = model(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]
        prediction = np.argmax(prediction)
        if prediction==0:
            return True
    return False

if __name__ == "__main__":
    print(liveness(0))