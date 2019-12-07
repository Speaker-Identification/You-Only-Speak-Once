"""
I didn't extract features from the test set of LibriSpeech, the features extracted
from train-100 was split into train and test set into two separate folders.
This was again done to read them easily using torch vision's Dataset Folder
"""

import os
import shutil
from pathlib import Path

import numpy as np


def assert_out_dir_exists(root, index):
    dir_ = root + '/' + str(index)

    if not os.path.exists(dir_):
        os.makedirs(dir_)
        print('crated dir {}'.format(dir_))
    else:
        print('dir {} already exists'.format(dir_))

    return dir_


def train_test_split(root, test_size=0.05):
    # make two folders, train and test
    train_dir = root + '_train'
    test_dir = root + '_test'

    os.makedirs(train_dir)
    os.makedirs(test_dir)

    for label in os.listdir(root):
        files_iter = Path(root + '/' + label).glob('**/*.npy')
        files_ = [str(f) for f in files_iter]
        files_ = np.array(files_)

        assert_out_dir_exists(train_dir, label)
        assert_out_dir_exists(test_dir, label)

        choices = np.random.choice([0, 1], size=files_.shape[0], p=(1 - test_size, test_size))
        train_files = files_[choices == 0]
        test_files = files_[choices == 1]

        for train_sample in train_files:
            src = train_sample
            dest = train_dir + '/' + label + '/' + train_sample.split('/')[-1]
            print('copying file {} to {}'.format(src, dest))
            shutil.copyfile(train_sample, train_dir + '/' + label + '/' + train_sample.split('/')[-1])

        for test_sample in test_files:
            src = test_sample
            dest = test_dir + '/' + label + '/' + test_sample.split('/')[-1]
            print('copying file {} to {}'.format(src, dest))
            shutil.copyfile(test_sample, test_dir + '/' + label + '/' + test_sample.split('/')[-1])

        print('done for label: {}'.format(label))

    print('All done')


if __name__ == '__main__':
    train_test_split('fbanks')