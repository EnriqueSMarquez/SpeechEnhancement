import os
import subprocess
import librosa
import numpy as np
import time
from scipy.io import wavfile
from torch.autograd import Variable
from glob import glob

import torch
from torch.utils import data

"""
Audio data preprocessing for SEGAN training.
It provides:
    1. 16k downsampling (sox required)
    2. slicing and serializing
    3. verifying serialized data
"""


# specify the paths - modify the paths at will
data_path = '/ssd/esm1g14/segan'  # the base folder for dataset
clean_train_foldername = 'clean_trainset_56spk_wav'  # where original clean train data exist
noisy_train_foldername = 'noisy_trainset_56spk_wav'  # where original noisy train data exist
out_clean_train_fdrnm = 'clean_trainset_wav_16k'  # clean preprocessed data folder
out_noisy_train_fdrnm = 'noisy_trainset_wav_16k'  # noisy preprocessed data folder
ser_data_fdrnm = 'ser_data'  # serialized data folder

# def resynthesize_speech(x,saving_path):
#     wavfile.write(saving_path,x)

class AudioResynthesizer():
    def __init__(self,model,data_folder_path,saving_folder,transform):
        self.transform = transform
        self.model = model
        self.saving_folder = saving_folder
        self.data_folder_path = data_folder_path
        self.file_names = os.listdir(self.data_folder_path)
    def reconstruct_file(self,name=None,output_name='',saving_folder=None,file_split=2):
        if name == None:
            name = self.file_names[np.random.randint(len(self.file_names))].split('.')[0]
        if saving_folder == None:
            saving_folder = self.saving_folder
        print('RESINTHESYSING %s')%(saving_folder+name)
        file_names = glob(self.data_folder_path+name+'*')
        nb_slices = len(file_names)
        clean_data = []
        noisy_data = []
        full_noisy_data = []
        for slice_index in range(nb_slices):
            slice_name = name + '.wav_' +str(slice_index)+'.npy'
            slice_data = np.load(self.data_folder_path + slice_name)
            if len(clean_data) == 0:
                clean_data += [slice_data[0]]
                noisy_data += [slice_data[1]]
                full_noisy_data += [slice_data[1]]
            else:
                clean_data += [slice_data[0][-len(slice_data[0])/2-1::]]
                noisy_data += [slice_data[1][-len(slice_data[1])/2-1::]]
                full_noisy_data += [slice_data[1]]
        full_noisy_data = np.asarray(full_noisy_data).reshape(len(full_noisy_data),1,-1).astype(float)
        # batch_noisy_data = Variable(self.transform(np.asarray(full_noisy_data).reshape(len(full_noisy_data),1,-1).astype(float))).cuda()
        batches = []
        for i in range(file_split):
            batches += [full_noisy_data[i*len(full_noisy_data)/file_split:(i+1)*len(full_noisy_data)/file_split,:,:]]
        predicted_speech = []
        for batch in batches:
            predicted_speech += [self.model(Variable(self.transform(batch).cuda())).data.cpu().numpy()]
        # predicted_speech = np.asarray(predicted_speech)
        # predicted_speech = self.model(batch_noisy_data).data.cpu().numpy()
        # predicted_speech = predicted_speech.reshape(len(predicted_speech),-1)
        predicted_speech = np.concatenate(predicted_speech,axis=0)
        cropped_predicted_speech = [predicted_speech[0]]
        for x in predicted_speech[1::]:
            cropped_predicted_speech += [x[:,-x.shape[1]/2-1::]]
        cropped_predicted_speech = np.concatenate(cropped_predicted_speech,axis=1).reshape(-1,1)
        clean_data = np.concatenate(clean_data,axis=0)
        noisy_data = np.concatenate(noisy_data,axis=0)
        wavfile.write(saving_folder + name +'_clean.wav',rate=16000,data=clean_data.reshape(-1,1))
        wavfile.write(saving_folder + name +'_noisy.wav',rate=16000,data=noisy_data.reshape(-1,1))
        wavfile.write(saving_folder + name +'_predicted' + output_name + '.wav',rate=16000,data=cropped_predicted_speech)

class AudioSampleGenerator(data.Dataset):
    """
    Audio sample reader.
    Used alongside with DataLoader class to generate batches.
    see: http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
    """
    def __init__(self, data_folder_path=os.path.join(data_path, ser_data_fdrnm),transforms=None,target_transforms=None,train=True,train_split=0.75):
        if not os.path.exists(data_folder_path):
            raise Error('The data folder does not exist!')

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepaths = [os.path.join(data_folder_path, filename)
                for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.train_split = 0.75
        self.train = train
        self.set_dataset(self.train)

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.
        Args:
            batch_size(int): batch size
        Returns:
            ref_batch: reference batch
        """
        ref_filenames = np.random.choice(self.filepaths, batch_size)
        ref_batch = torch.from_numpy(np.stack([np.load(f) for f in ref_filenames]))
        return ref_batch

    def fixed_test_audio(self, num_test_audio):
        """
        Randomly chosen batch for testing generated results.
        Args:
            num_test_audio(int): number of test audio.
                Must be same as batch size of training,
                otherwise it cannot go through the forward step of generator.
        """
        test_filenames = np.random.choice(self.filepaths, num_test_audio)
        test_noisy_set = [np.load(f)[1] for f in test_filenames]
        # file names of test samples
        test_basenames = [os.path.basename(fpath) for fpath in test_filenames]
        return test_basenames, np.array(test_noisy_set).reshape(num_test_audio, 1, 16384)

    def set_dataset(self,train):
        self.train = train
        if self.train:
            self.filepaths = self.filepaths[0:int(len(self.filepaths)*self.train_split)]
        else:
            self.filepaths = self.filepaths[int(len(self.filepaths)*self.train_split)::]
        self.num_data = len(self.filepaths)
    def reset_data(self):
        self.filepaths = [os.path.join(data_folder_path, filename)
                for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)
    def __getitem__(self, idx):
        # get item for specified index
        y,x = np.load(self.filepaths[idx])
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        if self.transforms:
            x = self.transforms(x)
        if self.target_transforms:
            y = self.target_transforms(y)
        return x,y

    def __len__(self):
        return self.num_data
    def chop_data(self,rate=0.1):
        np.random.shuffle(self.filepaths)
        self.filepaths = self.filepaths[0:int(self.num_data*rate)]
        self.num_data = len(self.filepaths)

def data_verify():
    """
    Verifies the length of each data after preprocessing.
    """
    ser_data_path = os.path.join(data_path, ser_data_fdrnm)
    for dirname, dirs, files in os.walk(ser_data_path):
        for filename in files:
            data_pair = np.load(os.path.join(dirname, filename))
            if data_pair.shape[1] != 16384:
                print('Snippet length not 16384 : {} instead'.format(data_pair.shape[1]))
                break


def downsample_16k(data_path=data_path,clean_folder_name=clean_train_foldername,noisy_folder_name=noisy_train_foldername,out_clean_fdrnm=out_clean_train_fdrnm,out_noisy_fdrnm=out_noisy_train_fdrnm):
    """
    Convert all audio files to have sampling rate 16k.
    """
    # clean training sets
    if not os.path.exists(os.path.join(data_path, out_clean_fdrnm)):
        os.makedirs(os.path.join(data_path, out_clean_fdrnm))

    for dirname, dirs, files in os.walk(os.path.join(data_path, clean_folder_name)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            output_folderpath = os.path.join(data_path, out_clean_fdrnm)
            # use sox to down-sample to 16k
            print('Downsampling : {}'.format(input_filepath))
            completed_process = subprocess.check_call(
                    'sox {} -r 16k {}'
                    .format(input_filepath, os.path.join(output_folderpath, filename)),
                    shell=True)

    # noisy training sets
    if not os.path.exists(os.path.join(data_path, out_noisy_fdrnm)):
        os.makedirs(os.path.join(data_path, out_noisy_fdrnm))

    for dirname, dirs, files in os.walk(os.path.join(data_path, noisy_folder_name)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            output_folderpath = os.path.join(data_path, out_noisy_fdrnm)
            # use sox to down-sample to 16k
            print('Processing : {}'.format(input_filepath))
            completed_process = subprocess.check_call(
                    'sox {} -r 16k {}'
                    .format(input_filepath, os.path.join(output_folderpath, filename)),
                    shell=True)


def slice_signal(filepath, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size with [stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(filepath, sr=sample_rate)
    n_samples = wav.shape[0]  # contains simple amplitudes
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_path=data_path,clean_folder_16k=out_clean_train_fdrnm,noisy_folder_16k=out_noisy_train_fdrnm,output_folder=ser_data_fdrnm):
    """
    Serialize the sliced signals and save on separate folder.
    """
    start_time = time.time()  # measure the time
    window_size = 2 ** 14  # about 1 second of samples
    dst_folder = os.path.join(data_path, output_folder)
    sample_rate = 16000
    stride = 0.5

    if not os.path.exists(dst_folder):
        print('Creating new destination folder for new data')
        os.makedirs(dst_folder)

    # the path for source data (16k downsampled)
    clean_data_path = os.path.join(data_path, clean_folder_16k)
    noisy_data_path = os.path.join(data_path, noisy_folder_16k)

    # walk through the path, slice the audio file, and save the serialized result
    for dirname, dirs, files in os.walk(clean_data_path):
        if len(files) == 0:
            continue
        for filename in files:
            print('Splitting : {}'.format(filename))
            clean_filepath = os.path.join(clean_data_path, filename)
            noisy_filepath = os.path.join(noisy_data_path, filename)

            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_filepath, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_filepath, window_size, stride, sample_rate)

            # serialize - file format goes [origial_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(dst_folder, '{}_{}'.format(filename, idx)), arr=pair)

    # measure the time it took to process
    end_time = time.time()
    print('Total elapsed time for prerpocessing : {}'.format(end_time - start_time))
