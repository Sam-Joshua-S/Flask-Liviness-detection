import time
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import cv2
import mediapipe as mp
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'model', 'model.pth')

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []
        self.handbox = []

    def take_pic_to_trans(self,cap):
        success = False
        img_id=-2
        equ =False
        t_end = time.time() + 20
        while img_id!=8 and time.time() < t_end:
            success, img = cap.read()
            box,hands, img = self.findHands(img) 
            
            if box != [] and box[0]<box[1] and box[2]<box[3]:
                img_id+=1
                if img_id>0:
                    after_box = img[box[2]:box[3],box[0]:box[1]]
                    stretch_near = cv2.resize(after_box, (200, 200),interpolation = cv2.INTER_LINEAR)
                    grayimg = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)
                    equ = cv2.equalizeHist(grayimg)
                    success = True
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
        return equ,success
        


    def create_data(self,hand_id,img_id,box,img):
        after_box = img[box[2]:box[3],box[0]:box[1]]
        stretch_near = cv2.resize(after_box, (200, 200),interpolation = cv2.INTER_LINEAR)
        grayimg = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(grayimg)
        cv2.imwrite(f"{hand_id}/img{img_id}.jpg",equ)
   
    def findHands(self, img, draw=False, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                self.handbox = xList[17],xList[5], yList[9],yList[0]
                
                
                mage = cv2.line(img, (0,yList[9]), (0,yList[0]), (255,155,0),20)
                
                
                mage = cv2.line(img, (xList[5],0), (xList[17],0), (255,155,155),20)
                if draw:
                    
                    
                    self.mpDraw.draw_landmarks(img, handLms,
                                             self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                
                    
        return self.handbox,allHands, img



def main(hand_id=None,demo=True):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    img_id = -4
    while img_id<10:
        
        success, img = cap.read()
        stretch_near = img
        box,hands, img = detector.findHands(img) 
        
        if box != [] and box[0]<box[1] and box[2]<box[3] :
            img_id+=1
            if demo==False and img_id>0:
                detector.create_data(hand_id,img_id,box,img)
    
        img = cv2.flip(img, 1)
        cv2.waitKey(1)
        if img_id=='10':  
            cap.release()
            cv2.destroyAllWindows()
            break
        



#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


import os

def compare(x0,x1):
    x1 = np.fromstring(x1[2:-2], dtype='float64', sep=' ')
    x1 = x1.reshape(1, -1)
    x1 = torch.tensor(x1, dtype=torch.float32)
    x0 = torch.tensor(x0, dtype=torch.float32)
    try:
        euclidean_distance = F.pairwise_distance(x0,x1)
        if(euclidean_distance.item()<10):
            return True
        else:
            return False
    except:
        print("No hand found")

def create(cap):
    net= SiameseNetwork()
    output1=False
    net.load_state_dict(torch.load(data_path))
    detector2 = HandDetector(detectionCon=0.8, maxHands=2)
    x1,exist = detector2.take_pic_to_trans(cap)
    if exist:
        x1 = Image.fromarray(x1)
        transformation = transforms.Compose([transforms.Resize((100,100)),
                                        transforms.ToTensor()
                                        ])
        x1 = transformation(x1)
        x1=x1.reshape(1,1,100,100).float()
        output1, _ = net(x1, x1)
        output1 = output1.detach().numpy()
    else:
        output1 = False
    return output1

if '__main__'==__name__:
    a=create()
    b=create()
    print(compare(a,b))