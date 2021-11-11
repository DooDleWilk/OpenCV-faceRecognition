print('Importing librairies...')
import face_recognition
import cv2
import os, pickle
from PIL import Image, ExifTags
import numpy
print('Librairies imported!')

def loadImage(path):
    print('Loading image from', path)
    pil_image = Image.open(path).convert("RGB")
    img_exif = pil_image.getexif()
    ret={}
    orientation = 0

    if img_exif:
        for tag, value in img_exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            ret[decoded] = value
        if 'Orientation' in ret:
            orientation = ret['Orientation']
    
    if orientation == 8:
        pil_image = pil_image.rotate(90, Image.NEAREST, expand=1)
    elif orientation == 3:
        pil_image = pil_image.rotate(180, Image.NEAREST, expand=1)
    elif orientation == 6:
        pil_image = pil_image.rotate(270, Image.NEAREST, expand=1)

    return numpy.array(pil_image)

def resizeImage(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
   
Encodings=[]
Names=[]

image_dir='/home/doodle/Pictures/Known/'
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path=os.path.join(root, file)
        name=os.path.splitext(file)[0]
        
        personFace=loadImage(path)
        
        print('Encoding face of', name)
        personEncoding=face_recognition.face_encodings(personFace)[0]

        Encodings.append(personEncoding)
        Names.append(name)

with open('/home/doodle/Documents/Pickles/train.pkl', 'wb') as f:
    pickle.dump(Names, f)
    pickle.dump(Encodings, f)
