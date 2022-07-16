import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns

import scipy
from scipy.stats.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from numpy import nan
from numpy import isnan
import statsmodels.api as sm

#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from numpy import nan
from numpy import isnan
import statsmodels.api as sm

#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import operator
import csv
from itertools import chain





class Datapreprocessing(object):

  def __init__(self, file_path):
    self.dataset = pd.read_csv(file_path)
    self.dff = pd.DataFrame(self.dataset) #add krsi
    #print(self.dff)
    self.dff_list = self.dff.values.tolist()
    self.for_lastcol = pd.DataFrame(self.dataset) #for accesing last colum
    self.for_lastcol = list(self.for_lastcol)
    self.dff.columns.values[len(self.dff.columns)-1] = 'Class'
    print(len(self.dff.columns))
    self.dff = list(self.dff)
    #print('--------------')
    #print(self.dff_list)
    #print(len(self.dff_list))
    #self.dff[self.dff[-1]] = 'Class'
    self.file_path = file_path
    self.dataset = self.dataset.replace('?',np.nan).astype(float)
    self.dataset = self.dataset.fillna((self.dataset.median()))
    self.target = self.dff[-1] #last column of csv
    self.seriesObject = self.get_VIF(self.dataset, self.target)
    self.numeric_column_findout(self.dataset)

    

  def get_VIF(self, dataFrame, target):
    X = add_constant(dataFrame.loc[:, dataFrame.columns != target])
    seriesObject = pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1])] , index=X.columns,)
    return seriesObject

  def numeric_column_findout(self, dataset):
    self.num_cols = dataset._get_numeric_data().columns
    results = {}
    num_cols = list(self.num_cols)
    my_dict ={
    };
    i = 0
    for col in num_cols:
      r, p = pearsonr(dataset[self.dff[-1]], dataset[col]) # last column
      my_dict[num_cols[i]] = []
      i = i + 1
      my_dict[col].append(abs(r))
    sorted_my_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(1),reverse=True)) 
    keys = list(sorted_my_dict.keys())
    self.rscores = list(sorted_my_dict.values())
    self.flatten_list = list(chain.from_iterable(self.rscores))
    #print(self.flatten_list)
    key_class = keys[0]
    keys.remove(keys[0]) # varaible
    keys.append(key_class)
    with open(self.file_path, 'r') as infile, open('ProcessedDataset.csv', 'w') as outfile:
      fieldnames = keys
      writer = csv.DictWriter(outfile, fieldnames=fieldnames)
      writer.writeheader()
      for row in csv.DictReader(infile):
        writer.writerow(row)
    Processed_dataset = pd.read_csv('ProcessedDataset.csv')
    return Processed_dataset

  def r_scores(self):
    r = self.flatten_list
    del r[0]
    return r

  def column_rename(self,csv_file):
    self.csv_frame = pd.read_csv(csv_file)
    self.csv_frame = pd.DataFrame(self.csv_frame)
    self.csv_frame.columns.values[len(self.csv_frame.columns)-1] = 'Class'
    self.csv_frame.to_csv('ProcessedDataset.csv')
