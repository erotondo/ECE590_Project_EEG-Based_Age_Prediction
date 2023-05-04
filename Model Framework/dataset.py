import os

import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data

__all__ = ['CNNDataset','LSTMDataset']


class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, participants_ids, preproc_eeg_data_dir, condition, partipant_metadata):
        self.part_ids = participants_ids
        self.eeg_dir = preproc_eeg_data_dir
        self.condition = condition # in ['restEC','restEO']
        self.metadata = partipant_metadata

    def __len__(self):
        return len(self.part_ids)

    def __getitem__(self, idx):
        # subject = '12345678'
        subject = self.part_ids[idx]

        # Identify participant session to use
        sub_content = os.listdir(os.path.join(self.eeg_dir,'sub-'+subject))
        if len(sub_content) > 1: # Randomly choose a session if participant has multiple
            sub_sess = random.choice(sub_content)
        else:
            sub_sess = sub_content[0]
        sess_num = sub_sess[-1]

        # Get participant's age for the session to use
        sub_rows = self.metadata[self.metadata.participants_ID == 'sub-'+subject]
        if len(sub_content) > 1:
            row = sub_rows[sub_rows.sessID==int(sess_num)]
        else:
            row = sub_rows
        age = float(row.age) # The target

        # Identify .npy file with preprocessed eeg data
        sub_data_dir = os.listdir(os.path.join(self.eeg_dir,'sub-'+subject,sub_sess,'eeg'))
        target_file = [x for x in sub_data_dir if self.condition in x][0]

        # Get eeg channel data
        sub_data = np.load(os.path.join(self.eeg_dir,'sub-'+subject,sub_sess,'eeg',target_file), allow_pickle=True)
        eeg_data = sub_data['data'][:,0:26,:]

        data_dims = eeg_data.shape
        input_block = np.zeros((data_dims[0], 4, data_dims[1], data_dims[2]))

        input_block[:,0,:,:] = eeg_data
        for i, r in enumerate([6, 14, 21]):
            input_block[:,i+1,:,:] = np.roll(eeg_data, r, axis=1)
        
        return input_block, np.full(input_block.shape[0], age), ['cyclic permutation eeg channel block', 'participant age']

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, participants_ids, preproc_eeg_data_dir, condition, partipant_metadata):
        self.part_ids = participants_ids
        self.eeg_dir = preproc_eeg_data_dir
        self.condition = condition # in ['restEC','restEO']
        self.metadata = partipant_metadata

    def __len__(self):
        return len(self.part_ids)

    def __getitem__(self, idx):
        # subject = '12345678'
        subject = self.part_ids[idx]

        # Identify participant session to use
        sub_content = os.listdir(os.path.join(self.eeg_dir,'sub-'+subject))
        if len(sub_content) > 1: # Randomly choose a session if participant has multiple
            sub_sess = random.choice(sub_content)
        else:
            sub_sess = sub_content[0]
        sess_num = sub_sess[-1]

        # Get participant's age for the session to use
        sub_rows = self.metadata[self.metadata.participants_ID == 'sub-'+subject]
        if len(sub_content) > 1:
            row = sub_rows[sub_rows.sessID==int(sess_num)]
        else:
            row = sub_rows
        age = float(row.age) # The target

        # Identify .npy file with preprocessed eeg data
        sub_data_dir = os.listdir(os.path.join(self.eeg_dir,'sub-'+subject,sub_sess,'eeg'))
        target_file = [x for x in sub_data_dir if self.condition in x][0]

        # Get eeg channel data
        sub_data = np.load(os.path.join(self.eeg_dir,'sub-'+subject,sub_sess,'eeg',target_file), allow_pickle=True)
        eeg_data = sub_data['data'][:,0:26,:]
        
        return eeg_data, np.full(eeg_data.shape[0], age), ['eeg channel sequence', 'participant age']