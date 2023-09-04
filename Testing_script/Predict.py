from Testing_script.TestingDataframes import get_annotation_information
import os
from os import listdir
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import configparser
import sys
import os

import configparser
import sys
import os

config_file_path = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file_path)
for section in config.sections():
    your_folder_directory = config[section]['parameter8']

base_name, extension = os.path.splitext(config_file_path)
print(f"Config file name (without extension): {base_name}")
config.read(config_file_path)

def compute_metrics():

    prediction_folder = your_folder_directory+f'/generative_data_augmentation_gibbons_copy/Testing_script/ModelPredictions/{base_name}/'
    annotation_folder = your_folder_directory+'/generative_data_augmentation_gibbons_copy/Testing_script/Annotations/'
    audio_folder = your_folder_directory+'/generative_data_augmentation_gibbons/Testing_script/Audio/'
    
    def delete_elements_by_names(lst, names):
        lst[:] = [item for item in lst if item not in names]

    pred_files=listdir(prediction_folder)
    df_all_predictions = pd.DataFrame()
    predictions=[]
    annotations=[]
    
    names_to_delete = ['.DS_Store', '.DS_Store.svl', '.DS_S.svl','.DS_s.svl','.DS_S']
    delete_elements_by_names(pred_files, names_to_delete)

    for file in pred_files:
        file_name=file[0:-4]

        predicted_svl = get_annotation_information(audio_folder,prediction_folder,file_name, False)
        df_all_predictions = pd.concat([df_all_predictions, predicted_svl])
        df_all_predictions.to_csv('predictions.csv')

    audio_files = listdir(audio_folder)
    df_all_annotations = pd.DataFrame()
    predictions = []
    annotations = []

    names_to_delete = ['.DS_Store', '.DS_Store.svl', '.DS_S.svl','.DS_s.svl','.DS_S']
    delete_elements_by_names(audio_files, names_to_delete)
    
    for file in audio_files:
        file_name = file[0:-4]

        annotation_svl = get_annotation_information(audio_folder , annotation_folder , file_name , False)
        df_all_annotations = pd.concat([df_all_annotations , annotation_svl])
        df_all_annotations.to_csv('annotations.csv')

    df1 = df_all_annotations
    df2 = df_all_predictions

    TP = []
    FP = []
    FN = []
    TN = []

    def to_timestamp(dt):
        epoch = datetime.utcfromtimestamp(0)
        return int((dt - epoch).total_seconds())

    for _, row1 in df1.iterrows():
        
        for _, row2 in df2.iterrows():
            overlap_percentage = min(1.0, (min(row2['End'], row1['End']) - max(row2['Start'], row1['Start'])).total_seconds() / (row1['End'] - row1['Start']).total_seconds())
            #print(overlap_percentage)
            if overlap_percentage == 1 and row1['Label'] == 'gibbon':
                TP.append(row1)
            if overlap_percentage == 1 and row1['Label'] == 'no-gibbon':
                FP.append(row1)
            if overlap_percentage > 0 and overlap_percentage < 1 and row1['Label'] == 'gibbon':
                FP.append(row1)
            if overlap_percentage > 0 and overlap_percentage < 1 and row1['Label'] == 'no-gibbon':
                FP.append(row1)
            if overlap_percentage == 0 and row1['Label'] == 'no-gibbon':
                TN.append(row1)
            if overlap_percentage == 0 and row1['Label'] == 'gibbon':
                FN.append(row1)

    epsilon = 0.0005
    accuracy = (len(TP) + len(TN)) / (len(TP) + len(FP) + len(TN) + len(FN) + epsilon)
    precision = len(TP) / (len(TP) + len(FP) + epsilon)
    recall = len(TP) / (len(TP) + len(FN) + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    #for file_name in pred_files:
        #file_path = os.path.join(prediction_folder , file_name)
        #if os.path.isfile(file_path):  # Check if it's a file (not a subdirectory)
            #try:
                #os.remove(file_path)
                #print(f"Deleted: {file_path}")
            #except Exception as e:
                #print(f"Error deleting {file_path}: {e}")

    return accuracy, precision, recall, f1_score