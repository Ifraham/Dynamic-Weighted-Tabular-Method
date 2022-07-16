import math
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

np.set_printoptions(linewidth=np.inf)
# os.mkdir("/content/datasets")


class ImageDatasetCreation(object):

  def __init__(self, file_path):
    self.df = pd.read_csv(file_path)
    self.dff = pd.DataFrame(self.df)
    self.remove_column = ['Class']
    self.df_mod = self.df.drop(columns=self.remove_column, axis=1)

  def features_insertion(self,m,n,a,l,h,c,f,s_list,A,L,H,C,F,arr):
    i=0
    j=0
    flag=0
    for k in range(0,len(f)):
        for j in range(0, m):
            for i in range(0, n):
                if arr[j,i]==0:
                    flag=0
              
                if j+h[k]>m or i+l[k]>n:
                    continue
                for y in range(j, j+h[k]):
                    for x in range(i, i+l[k]):
                        if arr[y,x]!=0:
                            flag=1    
              
                if flag==1:
                    continue
              
                elif flag==0:        
                    for y in range(j, j+h[k]):
                        for x in range(i, i+l[k]):
                            arr[y,x]=f[k]             
    
                            #print('Feature No = {}'.format(f[k])
                          
                    break
                # F.append(k)
                # H.append(h[k])
                # L.append(l[k])
                # C.append(c[k])
                # A.append(a[k])
          
            else:
                continue
            break
            
    area = a.copy()
    length = l.copy()
    height = h.copy()
    character = c.copy()
    features = f.copy()

 
    rows = len(arr)
    columns = len(arr[0])

    #Features that were not in arrays List Creation
  
    for k in range(0,len(features)):
      for i in range(0,rows):
        for j in range(0,columns):
          if arr[i,j]==features[k]:
            s_list.append([j,i])
            ind = f.index(features[k])
            f.remove(features[k])        
            a.pop(ind)
            l.pop(ind)
            h.pop(ind)
            c.pop(ind)
        
            break
        else:
            continue
        break

    # print('Starting_List = {}'.format(s_list))
    # print('Height = {}'.format(h))
    # print('Character = {}'.format(c))
  

  def trim(self,m,n,a,l,h,c,f,arr):
    for i in range (0,len(f)):
      h[i] = h[i] - 1
      l[i] = l[i] - c[i]
      a[i] = l[i] * h[i]
    return h,l,a
  
  def info(s_list,m,n,a,l,h,c,f,arr):
    print('Starting_List = {}'.format(s_list))
    print('Height = {}'.format(h))
    print('Character = {}'.format(c))

  def excludedFeatures(self,m,n,s_list,h,arr):
    print('Excluded Features: ')
    print('Feature font= {}'.format(h))
    print('start point = {}'.format(s_list))
    # print('Length = {}'.format(l))
    # print('Height = {}'.format(h))
    # print('Area = {}'.format(a))

  def updateValues(m,n,a,l,h,c,f):
    A = a.copy()
    L = l.copy()
    H = h.copy()
    C = c.copy()
    F = f.copy()

  def array_traversal(self,arr,f,m,n):
    start_point = []
    feature_font = []
    #print(f)
    for i in range(0,len(f)):
      feature_font.append(0)
    for k in range(1,len(f)+1):
      p = 0
      for i in range(0, m):
        for j in range(0, n):
        
          #print(k)
          if arr[i,j] == k and p == 0:
            start_point.append([i,j])
            p = 1
          
            
            #feature_font.append(1)
      q = 0
      count = 0 # calculating fontsize
      for s in range(0,m):
        for t in range(0,n):
          if arr[t][s] == f[k-1] and q == 0:
            q = 1
            y = t
            while True:
              print(y)
              if arr[y][s] != f[k-1] or y >= 127:
                break
              y =y+1
              count+=1

            feature_font[k-1] = count
       
    
          
                  
    return start_point, feature_font
    
  def preinsertion(self,c,r,m,n,global_col_len):
    # r = [
    #     0.82,
    #     0.82,   # eta lagbe 1st module theke
    #     0.82,
    #     0.76,
    #     0.72,
    #     0.71,
    #     0.7,
    #     0.68,
    #     0.42
    #     ]
    total = sum(r)
    area = []
    #R is ratio of R scores

    for i in range(0, global_col_len):
        x = m*n*r[i]/total
        area.append(x)
    f = []
    for i in range(0, global_col_len):
        f.append(i+1)    
        
    #c = c # this will  comes from CSV
    # c = get_char_num(charnumber_df, 0)

 



    h = [] #Height of Box
    l = [] #Length of Box
    a = [] #Area of Box

    for i in range(0, len(c)):
        x = math.sqrt(area[i]/c[i])
        h.append(math.floor(x))

    for i in range(0, len(c)):
        y = area[i]/h[i]
        l.append(math.floor(y))
        a.append(math.floor(y)*h[i])
        



    arr =  np.zeros([m, n], dtype = int)



    A = []
    L = []
    H = h.copy()
    C = []
    F = f.copy()
    s_list = []



    count = 0
    while len(f)!=0:
    

        self.features_insertion(m,n,a,l,h,c,f,s_list,A,L,H,C,F,arr) ###FOR FURKAN - Try t bvo complete Function in next line. I will look at it after I wake up
        #updateValues(m,n,a,l,h,c,f)      #Starting List and others needs to be updated based on third set of variables declared just before this loop
        #info(s_list,m,n,a,l,h,c,f)
        
        h,l,a = self.trim(m,n,a,l,h,c,f,arr)
        self.excludedFeatures(m,n,s_list,h,arr)
        

        #findStartingList(m,n,a,l,h,c,f)
        count = count+1
        if count>50:
            break
    print(arr)
    start_point, font_size = self.array_traversal(arr,F,m,n)
    
    return start_point, font_size
    
  def pre_preinsertion(self):
    feature_no = self.df_mod.shape[1]
    
    c = []
    dataset_list = self.dff.values.tolist()
    print(dataset_list)
    for i in range(1,len(dataset_list[1])):
      minn = 1
      for j in range(len(dataset_list)):
        num = str(dataset_list[j][i])
        #print(num)
        
        #print(num)
        count = len(num)
        # while num != 0:
        #   num = (num//10)
        #   count += 1
        if count >= minn:
          minn = count
      c.append(minn)
    return feature_no, c
