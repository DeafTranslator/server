import restoreModel as model
import socket
import base64
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

ipServer = "10.0.0.10"
portServer = 6789


serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

serverSocket.bind((ipServer, portServer))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# model.main()

while True:
    data, addr = serverSocket.recvfrom(58096)
    # newdata = data[:4] + data[lenght(data):]
    data1 = base64.b64decode(data)
    fh = open("./try/try.png", "wb")
    fh.write(data1)
    fh.close()
    # print(data)
    if(data1):
      # image = mpimg.imread('./try.png')
      # plt.imshow(image)
      # plt.show()
      print("message", data1)
      clientSocket.sendto(model.test().encode(), (addr))
      print('okk')

