import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn
from d2l import torch as d2l

dataset = pd.read_csv('data/heatflow_resid_test.csv')

# sns.heatmap(dataset.corr(), cmap="crest")
# plt.show()

labels = dataset['hfqc_resid']
train_data = dataset.drop(['hfqc_resid'], axis=1)

#Exploratory data analysis methods below

#normalize data
train_data = (train_data - train_data.mean()) / train_data.std()
train_data = train_data.values

#perform PCA on normalized data
