import restoreModel as model
import crop
import socket
import base64
import numpy as np
from PIL import Image
import cv2

ipServer = "0.0.0.0"
portServer = 5679


serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSocket.bind((ipServer, portServer))
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
  data, addr = serverSocket.recvfrom(60000)
  print(data)
  data1 = base64.b64decode(data)
  fh = open("./try.png", "wb")
  fh.write(data1)
  fh.close()
  newImage = cv2.imread('./try.png')
  # image1 = np.frombuffer(data1, dtype = np.uint8)
  # print("image from buffer", image1.shape)
  # newImage = cv2.imdecode(image1, cv2.IMREAD_GRAYSCALE)
  # print("image imdcode", newImage.shape)
  # newImage = cv2.resize(newImage, (400,400), interpolation = cv2.INTER_CUBIC)
  # print("image imdcode", newImage.shape)

  if(data1):
    print("message", newImage)
    clientSocket.sendto((crop.editImg(newImage)).encode(), (addr))
    print('okk')
  #   print('okk')

