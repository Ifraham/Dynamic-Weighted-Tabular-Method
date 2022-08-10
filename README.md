# A Dynamic Weighted Tabular Method for Convolutional Neural Networks
The Dynamic Weighted Tabular Method is a novel technique for applying Convolutional Neural Networks on tabular data. The method uses statistical techniques to compute the relevance of the features to the class. Based on the relevance/importance level of the features they are assigned pixel positions in the image. The features receive image canvas space based on the ratio of their importance to the class. The images are then fed into CNNs for classification. View the paper for more details - [PDF](https://arxiv.org/pdf/2205.10386.pdf), [Arxiv Paper](https://arxiv.org/abs/2205.10386)

## Citation Request: Kindly cite the below mentioned paper if you use this technique.

Bibtex - @article{iqbal2022dynamic,
  title={A Dynamic Weighted Tabular Method for Convolutional Neural Networks},
  author={Iqbal, Md Ifraham and Mukta, Md and Hossain, Saddam and Hasan, Ahmed Rafi},
  journal={arXiv preprint arXiv:2205.10386},
  year={2022}
}

MLA Citation - Iqbal, Md Ifraham, et al. "A Dynamic Weighted Tabular Method for Convolutional Neural Networks." arXiv preprint arXiv:2205.10386 (2022).

## How to Use:
**Instructions**

Before application please make sure that the Dataset is viable for DWTM application:
- Ensure that there are no missing values
- Ensure that the independent variable (Class) is the final column in the dataset
- Ensure that your label/prediction column of the dataset is named "Class"


Use the *DWTM_Implement_Image_Creation.ipynb* or alternatively follow these steps for successful implementation of the DWTM. For applying Steps 2-5, run the cells mentioned below. The notebook is numbered to make it easier to follow:

Upload the following python files from the directory *DWTM* to working directory and import the classes within them:

- 1.1 Data_Processing_Numerical.py
- 1.2 Data_Processing_Categorical.py
- 2.1 Image_Canvas_Creation.py
- 3.1 Image_Generate.py

```python
!git clone https://github.com/AnonymousCIKM1/DWTM.git
%cd /content/DWTM/DWTM
from DWTM import*
from Data_Processing_Categorical import Datapreprocessing as DataPreprocessingCategorical
from Data_Processing_Numerical import Datapreprocessing as DataPreprocessingNumerical
from Image_Canvas_Creation import ImageDatasetCreation 
from Image_Generate import ImageGenerate
```

Process your input Dataset to create a Processed Dataset. Save the Processed Dataset in the working directory.

```python

data_path = 'enter your data path'
z = 1

PD = DataPreprocessingNumerical(data_path)
#print(yo)
r = PD.r_scores()
print(r)
#yo.column_rename('/content/ProcessedDataset.csv')
```

-For datasets with only Numerical Data - Use 1.1 Data_Processing_Numerical.py

-For datasets containing Categorical Data - Use 1.2 Data_Processing_Categorical.py



Use the Processed Dataset to successfully divide the Canvas Space. Use 2.1 Image_Canvas_Creation.py for Canvas Space Creation
```python
#Processed Dataset Path
ImageCanvasCreation = ImageDatasetCreation('/content/DWTM/DWTM/ProcessedDataset.csv') 

feature_no, c = ImageCanvasCreation.pre_preinsertion()

m = 128 #Change according to preference
n = 128  #Change according to preference
#Image size of 128 by 128 is being used

s_list, h_list = ImageCanvasCreation.preinsertion(c,r,m,n,feature_no)
print(s_list)
print(h_list)
```

Create the Image Dataset using the Processed Dataset and Canvas Space information. Use 3.1 Image_Generate.py to create Image Dataset
```python
import pandas as pd
#Path for Processed Dataset
im = ImageGenerate('/content/DWTM/DWTM/ProcessedDataset.csv')
csv_path = "/content/DWTM/DWTM/ProcessedDataset.csv"
csv_path2 = data_path
df2 = pd.read_csv(csv_path2)
if len(s_list)!=len(h_list):
  h_list.pop()
df = pd.read_csv(csv_path)
df.info()
for i in range(0,df.shape[0]):
  class_name = df.iloc[i]['Class']#.astype(int) 
  image_file_name =  str(i)
  #ccc,  data = count_chars_val(d)
  
  data = []
  for i, v in enumerate(df.loc[i].to_list()):
    if i!=2:
      data.append(v)

  img_height = 128 #Change according to preference, match with m and n of Part 3
  img_width = 128 #Change according to preference, match with m and n of Part 3
  test_img = im.image_generator(data,class_name=class_name, s_list=s_list, h_list=h_list, img_height=128, img_width=128, image_file_name=image_file_name)
```

Zip and Download the Image Dataset

Use an image classification codebase to obtain your results on the Image Dataset. If required you can use the Image_Classification_Pytorch.ipnyb to classify the Image dataset and obtain your results.

**Point to be noted**
For Step-2 , File 1.1 can be applied on all kinds of datasets, however it is highly recommended to use File 1.2 for application on datasets containing Categorical Data as 1.1 will lead to inaccurate calculation of the feature weights of the categorical variables.


