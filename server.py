# import restoreModel as model
import socket
# import base64
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

ipServer = "10.1.1.207"
portServer = 6789


serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

serverSocket.bind((ipServer, portServer))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# model.main()

while True:
    data, addr = serverSocket.recvfrom(8096)
    # newdata = data[:4] + data[lenght(data):]
    # data = base64.b64decode(data)
    # fh = open("./try.png", "wb")
    # fh.write(base64.b64decode(data))
    # fh.close()
    print(data)
    if(data):
      # image = mpimg.imread('./try.png')
      # plt.imshow(image)
      # plt.show()
      print("message", data)
      # model.test(data)
      fuck = 'Fuck you'
      clientSocket.sendto(fuck, (addr))
      print('okk')

