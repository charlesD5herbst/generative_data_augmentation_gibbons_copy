from itertools import product
from data_loader import load_data
from cs_composer import build_cs
from results.metrics import mean_metrics, std_metrics, config_var
from results.data_saver import save_results
from statistics import mean
from statistics import stdev
from models.SVAE import *
from models.DVAE import *
from models.CNNNetwork import *
from Testing_script.PredictionHelper import *
from Testing_script.Predict import compute_metrics
import time
import configparser
import sys
import os

config_file_path = sys.argv[1]
base_name, extension = os.path.splitext(config_file_path)
print(f"Config file name (without extension): {base_name}")

config = configparser.ConfigParser()
config.read(config_file_path)


#config = {
#    'sample_factors': [0.1],
#    'vae_models': ['deep_vae'],
#    'latent_dims': [8],
#    'cnn_models': ['mobile_net'],
#    'augment_factors': [0.8]
#}

#for (sample_factor, vae_model, latent_dim, cnn_model, augment_factor) in product(
#        config['sample_factors'], config['vae_models'], config['latent_dims'], config['cnn_models'],
#        config['augment_factors']
#):

for section in config.sections():
    sample_factor = int(config[section]['parameter1'])
    vae_model = config[section]['parameter4']
    latent_dim = int(config[section]['parameter3'])
    cnn_model = config[section]['parameter2']
    augment_factor = int(config[section]['parameter5'])
    your_folder_directory = config[section]['parameter8']
    print(your_folder_directory)
    

    #walltime_start = time.time()
    x_train_pos, y_train_pos, x_train_neg, y_train_neg = load_data(sample_factor)
    #print(x_train_pos.shape)
    
    model = shallow_vae(latent_dim) if vae_model == 'shallow_vae' else deep_vae(latent_dim)
    '''
    train_conditions = [
    [x_train_pos, f'/Users/charlherbst/generative_data_augmentation_gibbons/vae_checkpoint/checkpoint_pos/{base_name}/cp.ckpt'],
    [x_train_neg, f'/Users/charlherbst/generative_data_augmentation_gibbons/vae_checkpoint/checkpoint_neg/{base_name}/cp.ckpt']
    ]
    #start_time_vae = time.time()
    for x_train, checkpoint_path in train_conditions:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=vae_loss)
        model.fit(x=x_train, y=x_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[cp_callback])
    #end_time_vae = time.time()
    #elapsed_time_vae = end_time_vae - start_time_vae
    #print("Time of vae", elapsed_time_vae)

    #num_samples = int(x_train_pos.shape[0] * augment_factor)
    '''
    num_samples = augment_factor
    print("Number of vae samples to generate", num_samples)
    args = [num_samples, latent_dim, model, x_train_pos, y_train_pos, x_train_neg, y_train_neg, base_name]
    
    x_train, y_train = build_cs(args)

    networks = CNNNetwork()
    
    model = networks.custom_model() if cnn_model == 'custom_cnn' else networks.mobile_net()
    if cnn_model != 'custom_cnn':
        x_train = tf.expand_dims(x_train, axis=3, name=None)
        x_train = tf.repeat(x_train, 3, axis=3)

    checkpoint_path = your_folder_directory+f'/generative_data_augmentation_gibbons_copy/cnn_checkpoint/checkpoint/{base_name}/cp.ckpt'
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path , save_weights_only=True , verbose=1)

    temp_metrics = {
        'accuracy': [] ,
        'f1_score': [] ,
        'precision': [] ,
        'recall': []
    }
    start_time_cnn = time.time()
    for _ in range(2):
        model.fit(x=x_train, y=y_train, batch_size=2, epochs=5, verbose=1, callbacks = [callback])

        predict = PredictionHelper(cnn_model)
        predict.predict_all_test_files(True)

        accuracy, precision, recall, f1_score = compute_metrics()

        temp_metrics['accuracy'].append(accuracy)
        temp_metrics['f1_score'].append(precision)
        temp_metrics['precision'].append(recall)
        temp_metrics['recall'].append(f1_score)
    #end_time_cnn = time.time()
    #elapsed_time_cnn = end_time_cnn - start_time_cnn
    #print("Time of cnn", elapsed_time_vae)
        

    mean_metrics['accuracy'].append(mean(temp_metrics['accuracy']))
    mean_metrics['f1_score'].append(mean(temp_metrics['f1_score']))
    mean_metrics['precision'].append(mean(temp_metrics['precision']))
    mean_metrics['recall'].append(mean(temp_metrics['recall']))

    std_metrics['accuracy'].append(stdev(temp_metrics['accuracy']))
    std_metrics['f1_score'].append(stdev(temp_metrics['f1_score']))
    std_metrics['precision'].append(stdev(temp_metrics['precision']))
    std_metrics['recall'].append(stdev(temp_metrics['recall']))

    config_var['sample_factor'].append(sample_factor)
    config_var['augment_factor'].append(augment_factor)
    config_var['cnn'].append(cnn_model)
    config_var['vae'].append(vae_model)
    config_var['lat_dim'].append(latent_dim)

df = save_results(mean_metrics, std_metrics, config_var, base_name)
#walltime_end = time.time()
#walltime = walltime_end = walltime_start

#print("Overall time", walltime)
pd.set_option('max_columns', None)
print(df)

        
        
        
        
        
        
  