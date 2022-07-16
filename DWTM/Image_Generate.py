import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import os 
import sys
from random import seed
from random import random
import pandas as pd
import matplotlib.pyplot as plt
seed(1)
import time 

from tqdm import tqdm

#!gdown --id 1dMJS5IB08NGiwfhNopczKlQ5tmsB8AaO

import os

# os.mkdir("/content/ImageDataset")
# os.mkdir("/content/ImageDataset/2")
# os.mkdir("/content/ImageDataset/4")


class ImageGenerate(object):

  def __init__(self,file_path):
    self.df = pd.read_csv(file_path)
    self.dff = pd.DataFrame(self.df)
    self.unique_column = self.df.Class.unique()
    os.mkdir("/content/ImageDataset")
    #!gdown --id 1dMJS5IB08NGiwfhNopczKlQ5tmsB8AaO
    for i in range(0,len(self.unique_column)):
      os.mkdir("/content/ImageDataset/"+str(self.unique_column[i]))




  def image_generator(self,data=None,class_name=None, s_list=None, h_list=None, saving_path=None, img_width=None, img_height=None,image_file_name=None):
    image_font = h_list.copy()
    for i in range(0, len(h_list)): 
      image_font[i] = image_font[i]/22#In case of font1, multiplier is 10

    starting_point = np.array(s_list.copy())
    for i in range(0, len(s_list)):
      starting_point[i] = s_list[i][1], s_list[i][0]

    s_point = list(starting_point)
    print(type(s_point))
    for i in range(0, len(s_list)):
      s_point[i][1] = s_point[i][1] + h_list[i]
    ###
    # saving the images 
    image = np.zeros((128, 128), dtype ="uint8") # create a image in numpy 

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (s_list[0])


    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    for i in range(0, len(s_list)):
      xx = starting_point[i]
      org = tuple(starting_point[i]) #position of s_list
      image = cv2.putText(image, str(data[i]), org, font, image_font[i], color, thickness, cv2.LINE_4)
    plt.imshow(image)
    image = Image.fromarray(image)
    label_folder = os.path.join("/content/ImageDataset", str(class_name))
    if os.path.exists(label_folder):
      saving_path =label_folder+"/"+image_file_name+".png"
      image.save(saving_path)
      print("<succes> Image saving succes")
      return  saving_path
    else:
      os.mkdir(label_folder)
      saving_path =label_folder+"/"+image_file_name+".png"
      image.save(saving_path)
      print("<succes> Image saving succes")
      return  saving_path
    ## In future it will return a image that will be store in dataset folder as an images







