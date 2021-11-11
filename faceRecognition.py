print('Importing librairies...')
import face_recognition
import cv2
import pickle
import time
from faceObject import Face
from threading import Thread
print('Librairies imported!')

# ---------------------------------------

class vStream:
    def __init__(self,src,width,height):
        self.width=width
        self.height=height
        self.capture=cv2.VideoCapture(src)
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            _,self.frame=self.capture.read()
            if len(self.frame) > 0:
                self.frame2=cv2.resize(self.frame,(self.width,self.height))
    def getFrame(self):
        return self.frame2

# ---------------------------------------

def getRectangleCenter(x1, y1, x2, y2):
    return int(x1 + ((x2 - x1) / 2)), int(y1 + ((y2 - y1) / 2))

def isInArea(x1, y1, x2, y2, size):
    if x1 > x2 - size and x1 < x2 + size and y1 > y2 - size and y1 < y2 + size:
        return True
    else:
        return False

# ---------------------------------------

# Setting variables
flip = 2
dispW = 640
dispH = 360

knownEncodings = []
knownNames = []
faces = []

font = cv2.FONT_HERSHEY_SIMPLEX

ratio = 4
pixels = 32

# Reading pickle from previous picture training
with open('/home/jetson/Documents/Pickles/train.pkl', 'rb') as f:
    knownNames=pickle.load(f)
    knownEncodings=pickle.load(f)

# Setting RPi Cam as capture device
#camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
camSet='nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=vStream(camSet,dispW,dispH)

cv2.namedWindow('piCam')

# FPS counter
startTime=time.time()
dtav=0

myFrame = None

while True:
    try:
        myFrame=cam.getFrame()
    except:
        print('Frame not available!')
    
    if myFrame is not None:
        myFrameSmall = cv2.resize(myFrame, (0,0), fx = 1/ratio, fy = 1/ratio)
        myFrameRGB = cv2.cvtColor(myFrameSmall, cv2.COLOR_BGR2RGB)
        faceLocations=face_recognition.face_locations(myFrameRGB, model='cnn')

        if len(faces) != len(faceLocations):
            faces.clear()
            newEncodings = face_recognition.face_encodings(myFrameRGB, faceLocations)

            for (faceLocTop, faceLocRight, faceLocBottom, faceLocLeft), newEncoding in zip(faceLocations, newEncodings):
                name = 'Unknown'
                matches = face_recognition.compare_faces(knownEncodings, newEncoding)

                if True in matches:
                    name = knownNames[matches.index(True)]

                faceLocTop = faceLocTop * ratio
                faceLocLeft = faceLocLeft * ratio
                faceLocBottom = faceLocBottom * ratio
                faceLocRight = faceLocRight * ratio

                newFace = Face(name, (faceLocTop, faceLocRight, faceLocBottom, faceLocLeft), newEncoding)
                faces.append(newFace)
   
        else:
            for face, (faceLocTop, faceLocRight, faceLocBottom, faceLocLeft) in zip(faces, faceLocations):
                (faceTop, faceRight, faceBottom, faceLeft) = face.getLocation()
                
                faceTopTemp = int(faceTop / ratio)
                faceRightTemp = int(faceRight / ratio)
                faceBottomTemp = int(faceBottom / ratio)
                faceLeftTemp = int(faceLeft / ratio)
                
                faceCenterXLoc, faceCenterYLoc = getRectangleCenter(faceLocLeft, faceLocTop, faceLocRight, faceLocBottom)
                faceCenterXTemp, faceCenterYTemp = getRectangleCenter(faceLeftTemp, faceTopTemp, faceRightTemp, faceBottomTemp)
                if isInArea(faceCenterXTemp, faceCenterYTemp, faceCenterXLoc, faceCenterYLoc, pixels):
                    
                    faceLocTop = faceLocTop * ratio
                    faceLocRight = faceLocRight * ratio
                    faceLocBottom = faceLocBottom * ratio
                    faceLocLeft = faceLocLeft * ratio

                    faceLocTopAvg = int(((1 - (1 / ratio)) * faceTop) + ((1 / ratio) * faceLocTop))
                    faceLocRightAvg = int(((1 - (1 / ratio)) * faceRight) + ((1 / ratio) * faceLocRight))
                    faceLocBottomAvg = int(((1 - (1 / ratio)) * faceBottom) + ((1 / ratio) * faceLocBottom))
                    faceLocLeftAvg = int(((1 - (1 / ratio)) * faceLeft) + ((1 / ratio) * faceLocLeft))
                    
                    faces[faces.index(face)].setLocation((faceLocTopAvg, faceLocRightAvg, faceLocBottomAvg, faceLocLeftAvg))

                (faceTop, faceRight, faceBottom, faceLeft) = faces[faces.index(face)].getLocation()
                cv2.rectangle(myFrame, (faceLeft, faceTop), (faceRight, faceBottom), (255,0,0), 2)
                cv2.putText(myFrame, face.getName(), (faceLeft, faceTop - 6), font, .75, (0,0,255), 1)

        # FPS calculation
        dt = time.time() - startTime
        startTime = time.time()
        dtav = .9 * dtav + .1 * dt
        fps = 1 / dtav

        # Display FPS counter
        cv2.rectangle(myFrame,(0,0),(140,40),(0,0,255),-1)
        cv2.putText(myFrame,str(round(fps,1))+' fps',(0,25),font,.75,(0,255,255),2)

        # Show final frame
        cv2.imshow('piCam', myFrame)
        cv2.moveWindow('piCam', 0, 0)
    else:
        print('Waiting for frame...')

    if cv2.waitKey(1) == ord('q'):
        break

cam.capture.release()
cv2.destroyAllWindows()
exit(1)
