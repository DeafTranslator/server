import tensorflow as tf
import time
from datetime import timedelta
import random
import os
import glob
import plot as pl
#=====================
import dataset
import config as conf
import matplotlib.pyplot as plt
import base64
import numpy as np
#=====================

#data = dataset.read_train_sets(conf.train_path, conf.img_size, conf.classes, validation_size=conf.validation_size)
test_images, test_ids = dataset.read_test_set(conf.test_path2, conf.img_size)


# ids = []
# i = 1
# path = os.path.join(conf.model_path, '*meta')
# files = glob.glob(path)
# for fl in files:
#   flbase = os.path.basename(fl)
#   print(str(i) + '-) ' + flbase)
#   i += 1
#   ids.append(flbase)

# while True: 
#   valor = input('\nSelect a model: ')
#   if int(valor) > 0 and int(valor) <= len(ids):
#     break;
#   else:
#     print('\tYou must select a number')

# print('You selected a model ' + ids[int(valor)-1])

# model = ids[int(valor)-1]
# model = model.split('.meta', 1)

session = tf.Session()
saver = tf.train.import_meta_graph('/Users/jesuspereyra/Documents/server/model/model_epoch101iteration1515_0.332208.meta')
saver.restore(session, '/Users/jesuspereyra/Documents/server/model/model_epoch101iteration1515_0.332208')

graph = tf.get_default_graph()
final_tensor = graph.get_tensor_by_name("final_result:0")
start_times = time.time()
      

def test2(image1):
  # image1 = tf.image.decode_png(data)
  # image1 = tf.image.decode_png(image1)
  # plt.imshow(image1)
  # tf.reshape(image1, conf.img_size_flat)
  # image1 = tf.reshape(conf.img_size_flat)
  feed_dict = {"x:0": [image1]}
  values = session.run(final_tensor, feed_dict=feed_dict)
  maxValue = session.run(tf.argmax(values, dimension=1))
  percentege = session.run(final_tensor, feed_dict=feed_dict)
  print(percentege)
  print('\nPrediction: ', conf.classes[maxValue[0]])
  # print('Real: ', test_ids[i].split('-', 1)[0])
  time_difs = end_times - start_times
  print("Time: " + str(round(time_difs)))
  pl.plot_image(image1)
  print('\n')


def test():
  end_times = time.time()
  # i = random.randint(1, len(test_images))
  # print('index: ' + str(i))
  data = 'iVBORw0KGgoAAAANSUhEUgAAAZAAAAGQCAIAAAAP3aGbAAAAA3NCSVQFBgUzC42AAAANrUlEQVR4Ae3YMYoDQRRDwe1l73/lNmbBmRW/gXI0WImob8Tgc+/98SFAgMATBH6fUFJHAgQIvAUMlt8BAQKPEfj7ND3nfJ49ECBAICXw/+eVN6zUUZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVgsFLnUIYAgSVgsJaOjACBlIDBSp1DGQIEloDBWjoyAgRSAgYrdQ5lCBBYAgZr6cgIEEgJGKzUOZQhQGAJGKylIyNAICVw7r2pQsoQIEDgm4A3rG8yvidAICfwAmhLDBs8MiVsAAAAAElFTkSuQmCC'
  data = base64.b64decode(data)
  print(data)
  image1 = np.frombuffer(data, np.uint8)
  # image1 = np.fromstring(data, dtype=np.float64)
  # image1 = test_images[0]
  image1 = image1.reshape(conf.img_size_flat)
  feed_dict = {"x:0": [image1]}
  values = session.run(final_tensor, feed_dict=feed_dict)
  maxValue = session.run(tf.argmax(values, dimension=1))
  percentege = session.run(final_tensor, feed_dict=feed_dict)
  print(percentege)
  print('\nPrediction: ', conf.classes[maxValue[0]])
  # print('Real: ', test_ids[i].split('-', 1)[0])
  time_difs = end_times - start_times
  print("Time: " + str(round(time_difs)))
  pl.plot_image(image1)
  print('\n')
# def main():
#   with tf.Session() as session:
#     saver = tf.train.import_meta_graph('/Users/jesuspereyra/Documents/server/model/model_epoch101iteration1515_0.332208.meta')
#     saver.restore(session, '/Users/jesuspereyra/Documents/server/model/model_epoch101iteration1515_0.332208')
                
#     graph = tf.get_default_graph()
#     final_tensor = graph.get_tensor_by_name("final_result:0")
#     start_times = time.time()
      
      # while True:
      #       cant = input('\n\nNumber of photos to analyze(0 - ' + str(len(test_images)) + '): ')
      #       if int(cant) > 0 and int(cant) <= len(test_images):
      #             break
      #       else:
      #             print('\tYou must select a number between 0 and ' + str(len(test_images))) 

      # print('\n')
      # while int(cant) > 0:
      #       test(session, final_tensor)
      #       cant = int(cant) - 1
