"""
This script extracts filter banks from audio files. Audio files are split
into frames of 25 ms and 64 F banks are extracted from each frame.
64 such frames are grouped together to create a sample which is a
64 x 64 matrix. Each matrix is saved as a .npy file into the output folder.
Samples from different speakers are in different folders and can be easily read
by torchvision's DatasetFolder.
"""

import os
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import python_speech_features as psf

BASE_PATH = 'LibriSpeech'
OUTPUT_PATH = 'fbanks'
np.random.seed(42)


def read_metadata():
    with open(BASE_PATH + '/SPEAKERS.TXT', 'r') as meta:
        data = meta.readlines()

    data = data[11:]
    data = ''.join(data)
    data = data[1:]
    data = re.sub(' +|', '', data)
    data = StringIO(data)

    speakers = pd.read_csv(data, sep='|', error_bad_lines=False)

    # This is using just the train clean 100 part. Update this line to extract from
    # train clean 360 or include both 100 and 360
    speakers_filtered = speakers[(speakers['SUBSET'] == 'train-clean-100')]
    speakers_filtered = speakers_filtered.copy()
    speakers_filtered['LABEL'] = speakers_filtered['ID'].astype('category').cat.codes
    speakers_filtered = speakers_filtered.reset_index(drop=True)
    return speakers_filtered


def get_fbanks(audio_file):

    def normalize_frames(signal, epsilon=1e-12):
        return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])

    y, sr = librosa.load(audio_file, sr=None)
    assert sr == 16000

    trim_len = int(0.25 * sr)
    if y.shape[0] < 1 * sr:
        # if less than 1 seconds, don't use that audio
        return None

    y = y[trim_len:-trim_len]

    # frame width of 25 ms with a stride of 10 ms. This will have an overlap of 15s
    filter_banks, energies = psf.fbank(y, samplerate=sr, nfilt=64, winlen=0.025, winstep=0.01)
    filter_banks = normalize_frames(signal=filter_banks)

    filter_banks = filter_banks.reshape((filter_banks.shape[0], 64, 1))
    return filter_banks


def assert_out_dir_exists(index):
    dir_ = OUTPUT_PATH + '/' + str(index)

    if not os.path.exists(dir_):
        os.makedirs(dir_)
        print('crated dir {}'.format(dir_))
    else:
        print('dir {} already exists'.format(dir_))

    return dir_


def main():
    speakers = read_metadata()

    print('read metadata from file, number of rows in in are: {}'.format(speakers.shape))
    print('numer of unique labels in the dataset is: {}'.format(speakers['LABEL'].unique().shape))
    print('max label in the dataset is: {}'.format(speakers['LABEL'].max()))
    print('number of unique index: {}, max index: {}'.format(speakers.index.shape, max(speakers.index)))

    for index, row in speakers.iterrows():
        subset = row['SUBSET']
        id_ = row['ID']
        dir_ = BASE_PATH + '/' + subset + '/' + str(id_) + '/'

        print('working for id: {}, index: {}, at path: {}'.format(id_, index, dir_))

        files_iter = Path(dir_).glob('**/*.flac')
        files_ = [str(f) for f in files_iter]

        index_target_dir = assert_out_dir_exists(index)

        sample_counter = 0

        for f in files_:
            fbanks = get_fbanks(f)
            num_frames = fbanks.shape[0]

            # sample sets of 64 frames each
            file_sample_counter = 0
            start = 0
            while start < num_frames + 64:
                slice_ = fbanks[start:start + 64]
                if slice_ is not None and slice_.shape[0] == 64:
                    assert slice_.shape[0] == 64
                    assert slice_.shape[1] == 64
                    assert slice_.shape[2] == 1
                    np.save(index_target_dir + '/' + str(sample_counter) + '.npy', slice_)

                    file_sample_counter += 1
                    sample_counter += 1

                start = start + 64

            print('done for index: {}, Samples from this file: {}'.format(index, file_sample_counter))

        print('done for id: {}, index: {}, total number of samples for this id: {}'.format(id_, index, sample_counter))
        print('')

    print('All done, YAY!, look at the files')


if __name__ == '__main__':
    main()
