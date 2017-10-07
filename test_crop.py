import cv2
import os
import glob
import numpy as np
import crop

train_path = 'C:\\Users\\Juan Graciano\\Desktop\\Imagenes\\Nati videos\\cropTape'

classesAlph = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
classesNum = ['0','1','2','3','4','5','6','7','8','9']
classes = classesNum

frame = None

k = 0

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def readVideo(video, fld):
    global frame
    cap = cv2.VideoCapture(video) 
    i = 0
    while( cap.isOpened() ) :
        ret,frame = cap.read()

        if ret is not True:
            break

        rotated = rotate_bound(frame, 90)

        frame = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("canny", cv2.Canny(frame, 127, 255))
        out = crop.editImg(frame)
        cv2.imshow("out", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def readFolder():
    global frame
    print('Reading image')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld,'*g')
        files = glob.glob(path)
        for fl in files:
            frame = cv2.imread(fl)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("canny", cv2.Canny(frame, 127, 255))
            out = crop.editImg(frame)
            cv2.imshow("out", out)
            cv2.waitKey(0)

readFolder()
