import cv2
import numpy as np

WIDTH = 334
HEIGHT = 334

def drawContour(frame):
        # grey = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        value = (15, 15)
        blurred = cv2.GaussianBlur(frame, value, 0)
        _, thresh1 = cv2.threshold(blurred,50,255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        height, width, _ = frame.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        # computes the bounding box for the contour, and draws it on the frame,
        hand
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
            if w > 90 and h > 90:
                bMax = bMin = 10
                if y - bMin <= 0 and x - bMin <= 0:
                    bMin = -1
                if y+h + bMax >= frame.shape[0] and x+w + bMax >= frame.shape[1]:
                    bMax = 0
                hand = frame[int(y-bMin):int(y+h+bMax), int(x-bMin):int(x+w+bMax)]
             	

        return hand

def mergeImage(frame, width, height):
 
    # Adjusting size
    if frame.shape[0] > height:
        hy = height/frame.shape[0]
        hx = frame.shape[1]*hy
        frame = cv2.resize(frame, (int(hx), int(height)), interpolation = cv2.INTER_CUBIC)
        print("Change y", frame.shape)
    if frame.shape[1] > width:
        hx = width/frame.shape[1]
        hy = frame.shape[0]*hx
        frame = cv2.resize(frame, (int(width), int(hy)), interpolation = cv2.INTER_CUBIC)
        print("Change x", frame.shape)
    
    # Mask
    merge = np.zeros((int(width), int(height), 3))
    
    # White mask
    merge.fill(255)
    
    # Putting the image in the middle
    centerMerge = width / 2
    centerFrame = frame.shape[1] / 2
    x_offset = centerMerge - centerFrame
    y_offset = height - frame.shape[0]

    # Merge
    merge[int(y_offset):int(y_offset+frame.shape[0]), int(x_offset):int(x_offset+frame.shape[1])] = frame

    return merge

def makeCanny_WB(frame, imgCanny):
    imgWB = cv2.threshold(frame,58,255,cv2.THRESH_BINARY)
    out = imgWB[1] + imgCanny
    return out

def editImg(img):
    imgCanny = cv2.Canny(img.copy(), 80, 255)
    frame = drawContour(img.copy())
    frame = mergeImage(frame, WIDTH, HEIGHT)
    outFrame = makeCanny_WB(frame, imgCanny)

    anything = 
    
    return anything
