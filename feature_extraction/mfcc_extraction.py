import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

BASE_PATH = 'LibriSpeech'
SAMPLE_SIZE = 10
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
    speakers_filtered = speakers[(speakers['SUBSET'] == 'train-clean-100') | (speakers['SUBSET'] == 'train-clean-360')]
    speakers_filtered = speakers_filtered.copy()
    speakers_filtered['LABEL'] = speakers_filtered['ID'].astype('category').cat.codes
    speakers_filtered = speakers_filtered.reset_index(drop=True)
    return speakers_filtered


def get_mfccs(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    assert sr == 16000

    # frame width of 25 ms with a stride of 15 ms. This will have an overlap of
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=13, hop_length=int(0.015 * sr), n_fft=int(0.025 * sr))
    mfcc_delta = librosa.feature.delta(mfccs, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfccs, order=2)
    result = np.append(mfccs, mfcc_delta, axis=0)
    result = np.append(result, mfcc_delta_delta, axis=0)
    result = result.T

    if result.shape[0] > SAMPLE_SIZE:
        indices = np.random.choice(result.shape[0], SAMPLE_SIZE, replace=False)
        result = result[indices]

    return result


def main():
    features = None
    labels = None
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

        for f in files_:
            mfccs = get_mfccs(f)
            if features is None:
                features = mfccs
                labels = np.full(mfccs.shape[0], index, dtype=np.uint8)
            else:
                features = np.append(features, mfccs, axis=0)
                labels = np.append(labels, np.full(mfccs.shape[0], index, dtype=np.uint8), axis=0)

        print('done for id: {}, index: {}, current shape of features: {}, labels: {}'.format(id_, index, features.shape, labels.shape))
        print('max label right now is: {}'.format(np.max(labels)))
        print('')

        if features.shape[0] >= 1000:
            break

    print('All done, writing to file')

    np.save('features.npy', features)
    np.save('labels.npy', labels)

    print('All done, YAY!, look at the files')


if __name__ == '__main__':
    main()
