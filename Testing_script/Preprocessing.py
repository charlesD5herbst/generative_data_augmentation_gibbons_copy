import glob, os
import numpy as np
import random
import librosa.display
import librosa
from xml.dom import minidom
from scipy import signal
from random import randint
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import math
import datetime
from AnnotationReader import *


class Preprocessing:
    
    def __init__(self, species_folder, lowpass_cutoff, 
                 downsample_rate, nyquist_rate, segment_duration, 
                 positive_class, background_class,
                 augmentation_amount_positive_class,
                 augmentation_amount_background_class,
                 n_fft, hop_length, n_mels, f_min, f_max, file_type, audio_extension):
        self.species_folder = species_folder
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.augmentation_amount_positive_class = augmentation_amount_positive_class
        self.augmentation_amount_background_class = augmentation_amount_background_class
        self.positive_class = positive_class
        self.background_class = background_class
        self.audio_path = self.species_folder + '/Audio/'
        self.annotations_path = self.species_folder + '/Annotations/'
        self.saved_data_path = self.species_folder + '/Saved_Data/'
        self.training_files = self.species_folder + '/DataFiles/TestingFiles.txt'
        self.n_ftt = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.file_type = file_type
        self.audio_extension = audio_extension
        
    def update_audio_path(self, audio_path):
        self.audio_path = self.species_folder + '/'+ audio_path
        
    def read_audio_file(self, file_name):
        '''
        file_name: string, name of file including extension, e.g. "audio1.wav"
        
        '''
        # Get the path to the file
        audio_folder = os.path.join(file_name)
        
        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)
        
        return audio_amps, audio_sample_rate
    
    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        '''
        Downsample an audio file to a given new sample rate.
        amplitudes:
        original_sr:
        new_sample_rate:
        
        '''
        return librosa.resample(amplitudes, 
                                original_sr, 
                                new_sample_rate, 
                                res_type='kaiser_fast'), new_sample_rate

    def convert_single_to_image(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_ftt,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        
        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
        
    
        # 3 different input
        return S1

    def convert_all_to_image(self, segments):
        '''
        Convert a number of segments into their corresponding spectrograms.
        '''
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment))

        return np.array(spectrograms)
    
    def add_extra_dim(self, spectrograms):
        '''
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.
        '''
        spectrograms = np.reshape(spectrograms, 
                                  (spectrograms.shape[0],
                                   spectrograms.shape[1],
                                   spectrograms.shape[2],1))
        return spectrograms
    
    def time_shift(self, audio, shift):
        '''
        Shift ampltitude values (to the right) by a random value.
        Values are wrapped back to the left.
        '''

        augmented = np.zeros(len(audio))
        augmented [0:shift] = audio[-shift:]
        augmented [shift:] = audio[:-shift]
        return augmented
    
    def augment_single_segment(self, segment, label, meta_time, verbose):
        '''
        Augment a segment of amplitude values a number of times (pre-defined).
        Augmenting is done by applying a time shift.
        '''
    
        augmented_segments = []
        augmented_metadata = []
        augmented_labels = []
        
        if label in self.positive_class:
            augmentation_amount = self.augmentation_amount_positive_class
        else:
            augmentation_amount = self.augmentation_amount_background_class

        for i in range (0, augmentation_amount):

            if verbose:
                print ('augmentation iteration', i)
                print ('----------------------------')

            # Randomly select amount to shift by
            random_time_point_segment = randint(1, len(segment)-1)

            # Time shift
            segment = self.time_shift(segment, random_time_point_segment)

            # Append the augmented segments
            augmented_segments.append(segment)

            # Duplicate the metadata for each augmentation example
            # as the metadata should not change
            augmented_metadata.append(meta_time)
            
            # Append the label (a multiclass problem can be converted into
            # one-hot encoded vectors at a later stage)
            augmented_labels.append(label)

        return augmented_segments, augmented_metadata, augmented_labels
    
    def getXY(self, audio_amplitudes, start_sec, annotation_duration_seconds, 
        label, file_name_no_extension, verbose):
        '''
        Extract a number of segments based on the user-annotations.
        If possible, a number of segments are extracted provided
        that the duration of the annotation is long enough. The segments
        are extracted by shifting by 1 second in time to the right.
        Each segment is then augmented a number of times based on a pre-defined
        user value.
        '''

        if verbose == True:
            print ('start_sec', start_sec)
            print ('annotation_duration_seconds', annotation_duration_seconds)
            print ('self.segment_duration ', self.segment_duration )
            
        X_augmented_segments = []
        X_meta_augmented_segments = []
        Y_augmented_labels = []
            
        # Calculate how many segments can be extracted based on the duration of
        # the annotated duration. If the annotated duration is too short then
        # simply extract one segment. If the annotated duration is long enough
        # then multiple segments can be extracted.
        if annotation_duration_seconds-self.segment_duration < 0:
            segments_to_extract = 1
        else:
            segments_to_extract = annotation_duration_seconds-self.segment_duration+1
            
        if verbose:
            print ("segments_to_extract", segments_to_extract)
            
        if label in self.background_class:
            if segments_to_extract > 15:
                segments_to_extract = 15
            
        for i in range (0, segments_to_extract):
            if verbose:
                print ('Semgnet {} of {}'.format(i, segments_to_extract-1))
                print ('*******************')
                
            # Set the correct location to start with.
            # The correct start is with respect to the location in time
            # in the audio file start+i*sample_rate
            start_data_observation = start_sec*self.downsample_rate+i*(self.downsample_rate)
            # The end location is based off the start
            end_data_observation = start_data_observation + (self.downsample_rate*self.segment_duration)
            
            # This case occurs when something is annotated towards the end of a file
            # and can result in a segment which is too short.
            if end_data_observation > len(audio_amplitudes):
                continue

            # Extract the segment of audio
            X_audio = audio_amplitudes[start_data_observation:end_data_observation]

            # Determine the actual time for the event
            start_time_seconds = start_sec + i

            # Extract some meta data (here, the hour at which the vocalisation event occured)
            meta_time = self.get_metadata(file_name_no_extension, start_time_seconds)

            if verbose == True:
                print ('start frame', start_data_observation)
                print ('end frame', end_data_observation)
            
            # Augment the segment a number of times
            X_augmented, X_meta_augmented, Y_augmented = self.augment_single_segment(X_audio, label, meta_time, verbose)
            
            # Extend the augmented segments and labels (and the metadata)
            X_augmented_segments.extend(X_augmented)
            X_meta_augmented_segments.extend(X_meta_augmented)
            Y_augmented_labels.extend(Y_augmented)

        return X_augmented_segments, X_meta_augmented_segments, Y_augmented_labels
    
    def save_data_to_pickle(self, X, Y):
        '''
        Save all of the spectrograms to a pickle file.
        
        '''
        outfile = open(os.path.join(self.saved_data_path, 'X-pow.pkl'),'wb')
        pickle.dump(X, outfile, protocol=4)
        outfile.close()
        
        outfile = open(os.path.join(self.saved_data_path, 'Y-pow.pkl'),'wb')
        pickle.dump(Y, outfile, protocol=4)
        outfile.close()
        
    def load_data_from_pickle(self):
        '''
        Load all of the spectrograms from a pickle file
        
        '''
        infile = open(os.path.join(self.saved_data_path, 'X.pkl'),'rb')
        X = pickle.load(infile)
        infile.close()
        
        infile = open(os.path.join(self.saved_data_path, 'Y.pkl'),'rb')
        Y = pickle.load(infile)
        infile.close()

        return X, Y


    def get_metadata(self, filename, time_to_add):

        '''
            Get some metadata from the file name.

            In this case, get the hour at which the particular vocalisation
            event occured. This is computed using the timestamp in the .wav
            file name and by adding on the time at which the event occured
            within the file. All vocalisation events are included, i.e.
            presence and absence events.

            filename: string, file name
            time_to_add: integer, seconds for which event occured

            Assumption: the filename has the following format: 
            HGSM3XX_0+1_YYYYMMDD_HHMMSS
            Where XX = location (A, B, C, AB, ...)

            Return: Integer (0, 1, ..., 23)

        '''

        print('meta')
        print('*********')
        print ('filename',filename)


        # Find the position of 'SM3' to determine the location name
        index_SM3 = filename.find('SM3')+3

        # Find the position of the first underscore
        index_first_underscore = filename.find('_')

        # Extract location name
        location = filename[index_SM3:index_first_underscore]

        # Find the position of the second underscore
        index_second_underscore = filename.find('0+1')+4

        # Find the position of the last underscore
        index_last_underscore = filename.rfind('_')

        # Get the date component from the string
        date_found = filename[index_second_underscore:index_last_underscore]

        # Get the time component from the string
        time = filename[index_last_underscore+1:]

        print('location',location)
        print('date found',date_found)
        print ('time found',time)

        year = int(date_found[0:4])
        month = int(date_found[4:6])
        date = int(date_found[6:])
        hour = int(time[0:2])
        minutes = int(time[2:4])
        seconds = int(time[4:])

        date_and_time = datetime.datetime(year, month, date, hour, minutes, seconds)
        print ('original time',date_and_time)

        # Compute the new time by taking the starting time from the file name
        # and adding on the time of the vocalisation event.
        new_time = date_and_time + datetime.timedelta(seconds=time_to_add)
        print ('new time', new_time)
        print ('encoded time', new_time.hour)

        # Add any other logical operations here on how to encode the time component
        # example 1: if hour < 12 then return 'am', else return 'pm'
        # example 2: if date_and_time.hour >= 0 and date_and_time.hour < 4 return 'early'
        # Other things can also be returned, e.g. location, or any other metadata

        return new_time.hour

    def create_dataset(self, verbose):
        '''
        Create X and Y values which are inputs to a ML algorithm.
        Annotated files (.svl) are read and the corresponding audio file (.wav)
        is read. A low pass filter is applied, followed by downsampling. A 
        number of segments are extracted and augmented to create the final dataset.
        Annotated files (.svl) are created using SonicVisualiser and it is assumed
        that the "boxes area" layer was used to annotate the audio files.
        '''
        
        # Keep track of how many calls were found in the annotation files
        total_calls = 0

        # Initialise lists to store the X and Y values
        X_calls = []
        X_meta = []
        Y_calls = []
        
        if verbose == True:
            print ('Annotations path:',self.annotations_path+"*.svl")
            print ('Audio path',self.audio_path+"*.wav")
        
        # Read all names of the training files
        training_files = pd.read_csv(self.training_files, header=None)
        
        # Iterate over each annotation file
        for training_file in training_files.values:
            
            file = training_file[0]
            
            if self.file_type == 'svl':
                # Get the file name without paths and extensions
                file_name_no_extension = file
                #print ('file_name_no_extension', file_name_no_extension)
            if self.file_type == 'raven_caovitgibbons':
                file_name_no_extension = file[file.rfind('-')+1:file.find('.')]
                
            print ('Processing:',file_name_no_extension)
            
            reader = AnnotationReader(file, self.species_folder, self.file_type, self.audio_extension)

            # Check if the .wav file exists before processing
            if self.audio_path+file_name_no_extension+self.audio_extension  in glob.glob(self.audio_path+"*"+self.audio_extension):
                print('Found file')
                
                # Read audio file
                audio_amps, original_sample_rate = self.read_audio_file(self.audio_path+file_name_no_extension+self.audio_extension)
                print('Filtering...')
                # Low pass filter
                filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)

                print('Downsampling...')
                # Downsample
                amplitudes, sample_rate = self.downsample_file(filtered, original_sample_rate, self.downsample_rate)

                df, audio_file_name = reader.get_annotation_information()

                print('Reading annotations...')
                for index, row in df.iterrows():

                    start_seconds = int(round(row['Start']))
                    end_seconds = int(round(row['End']))
                    label = row['Label']
                    annotation_duration_seconds = end_seconds - start_seconds

                    # Extract augmented audio segments and corresponding binary labels
                    X_data, X_meta_data, y_data = self.getXY(amplitudes, start_seconds, 
                                                annotation_duration_seconds, label, 
                                                file_name_no_extension, verbose)

                    # Convert audio amplitudes to spectrograms
                    X_data = self.convert_all_to_image(X_data)

                    # Append the segments and labels
                    X_calls.extend(X_data)
                    X_meta.extend(X_meta_data)
                    Y_calls.extend(y_data)

        # Convert to numpy arrays
        X_calls, X_meta, Y_calls = np.asarray(X_calls), np.asarray(X_meta), np.asarray(Y_calls)
        
        # Add extra dimension
        X_calls = self.add_extra_dim(X_calls)

        return X_calls, X_meta, Y_calls

