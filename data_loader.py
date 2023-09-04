import numpy as np
import pickle
from sklearn.utils import shuffle
import pandas as pd
import configparser
import sys
import os

config_file_path = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file_path)
for section in config.sections():
    your_folder_directory = config[section]['parameter8']



def load_data(num_samples):
    with open(your_folder_directory+'/generative_data_augmentation_gibbons_copy/data/training_spectrograms_df_final.pkl' , 'rb') as g:
        df = pickle.load(g)

    
    num_pos_samples = int(num_samples/2)
    num_neg_samples = int(num_samples/2)
    train_pos = df.loc[df['index'] == 'gibbon'].sample(n=num_pos_samples)
    train_neg = df.loc[df['index'] == 'no-gibbon'].sample(n=num_neg_samples)

    y_train_pos = train_pos.iloc[: , 0]
    y_train_neg = train_neg.iloc[: , 0]

    x_train_pos = np.array(train_pos.drop(['index'] , axis=1)).reshape(-1 , 128 , 128 , 1).astype("float32")
    x_train_neg = np.array(train_neg.drop(['index'] , axis=1)).reshape(-1 , 128 , 128 , 1).astype("float32")

    return x_train_pos , y_train_pos , x_train_neg , y_train_neg












