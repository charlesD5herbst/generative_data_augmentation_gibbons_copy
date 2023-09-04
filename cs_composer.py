import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf


def build_cs(args):
    num_samples, latent_dim, model, x_train_pos, y_train_pos, x_train_neg, y_train_neg, base_name = args

    checkpoint_pos = f'/Users/charlherbst/generative_data_augmentation_gibbons/vae_checkpoint/checkpoint_pos/{base_name}/cp.ckpt'
    checkpoint_neg = f'/Users/charlherbst/generative_data_augmentation_gibbons/vae_checkpoint/checkpoint_neg/{base_name}/cp.ckpt'

    model.load_weights(checkpoint_pos)
    random_normal = np.random.normal(0, 1, size=(num_samples , 1 , latent_dim))
    syn_pos_samples = np.array(model.decoder(random_normal))

    model.load_weights(checkpoint_neg)
    random_normal = np.random.normal(0, 1, size=(num_samples , 1 , latent_dim))
    syn_neg_samples = np.array(model.decoder(random_normal))

    if num_samples > 0:
        x = np.concatenate((x_train_pos, x_train_neg, syn_pos_samples, syn_neg_samples), axis=0)
        y = y_train_pos.tolist() + y_train_neg.tolist() + (['gibbon'] * num_samples) + (['no-gibbon'] * num_samples)
    else:
        x = np.concatenate((x_train_pos, x_train_neg), axis=0)
        y = y_train_pos.tolist() + y_train_neg.tolist()

    y = np.array(y)
    x = x.reshape((x.shape[0], 128*128))
    df = pd.DataFrame(x, y).reset_index()
    df = shuffle(df)

    df['index'] = df['index'].map({'gibbon': 1 , 'no-gibbon': 0})
    y_train = df.iloc[:, 0]
    y_train = tf.one_hot(y_train, 2)
    y_train = np.array(y_train).astype("float32")
    x_train = df.iloc[:, 1::]
    x_train = np.array(x_train).reshape(x_train.shape[0], 128, 128)
    x_train = np.expand_dims(x_train, -1).astype("float32")
    x_train = x_train.reshape(-1, 128, 128, 1)

    augmented_x = x_train
    augmented_y = y_train

    return augmented_x, augmented_y