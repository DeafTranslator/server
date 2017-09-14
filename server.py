import restoreModel as model
import socket
import base64
import numpy as np
from PIL import Image
from 
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

ipServer = "10.0.0.10"
portServer = 6789


serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

serverSocket.bind((ipServer, portServer))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# model.main()

# while True:
#     data, addr = serverSocket.recvfrom(58096)
#     # newdata = data[:4] + data[lenght(data):]
#     data1 = base64.b64decode(data)
#     fh = open("./try/try.png", "wb")
#     fh.write(data1)
#     fh.close()
#     # print(data)
#     if(data1):
#       # image = mpimg.imread('./try.png')
#       # plt.imshow(image)
#       # plt.show()
#       print("message", data1)
#       clientSocket.sendto(model.test().encode(), (addr))
#       print('okk')

def write_png(buf, width, height):
  """ buf: must be bytes or a bytearray in Python3.x,
      a regular string in Python2.x.
  """
  import zlib, struct

  # reverse the vertical line order and add null bytes at the start
  width_byte_4 = width * 4
  raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
    for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

  def png_pack(png_tag, data):
    chunk_head = png_tag + data
    return (struct.pack("!I", len(data)) +
      chunk_head +
      struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

  return b''.join([
    b'\x89PNG\r\n\x1a\n',
    png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
    png_pack(b'IDAT', zlib.compress(raw_data, 9)),
    png_pack(b'IEND', b'')])

while True:
  data, addr = serverSocket.recvfrom(58096)
  # newdata = data[:4] + data[lenght(data):]

  data1 = base64.b64decode(data)

  # print(data1.shape)

  image = write_png(data, 400, 400)
  # with open("my_image.png", 'wb') as fd:
  #     fd.write(data)
  # Decoding
  # sImage = base64.decodestring(data)
  # im = Image.fromarray(Image)

  # String to array
  # receivedImage = np.Array(image, dtype = np.uint8)
  image1 = np.frombuffer(data1, dtype = np.uint8)
  print(image1.shape)
  # Reshape array
  # receivedImage = receivedImage.reshape(400,400, 3)

  # fh = open("./try/try.png", "wb")
  # fh.write(data1)
  # fh.close()
  model.test(image1)
  # print(data)
  if(data1):
    # image = mpimg.imread('./try.png')
    # plt.imshow(image)
    # plt.show()
    print("message", data1)
    # clientSocket.sendto(model.testImage(receivedImage).encode(), (addr))
    print('okk')

