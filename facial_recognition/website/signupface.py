import face_recognition
import cv2
import time
import numpy as np

def face_recog(camera=0):
    t_end = time.time() + 20
    vid = cv2.VideoCapture(camera)
    
    while time.time() < t_end:
        try:
            ret, frame = vid.read()
            rgb_img1=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Collecting Face Data",frame)
            cv2.waitKey(1)
            img_encoding1=face_recognition.face_encodings(rgb_img1)[0]
            print(img_encoding1.dtype)
            return img_encoding1
        except IndexError as e:
            pass
    return []
def face_recognizer(img1,img2):
    img2=np.fromstring(img2[1:-1], dtype='float64', sep=' ')
    result= face_recognition.compare_faces([img1], img2)
    return result[0]
if "__main__"==__name__:
    img1 = face_recog()
    img2 = face_recog()
    print(face_recognizer(img1,img2))
      