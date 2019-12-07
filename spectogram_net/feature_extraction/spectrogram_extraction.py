import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os, errno
from pathlib import Path
import uuid

root = 'LibriSpeech'
speakers = pd.read_csv(root + '/speakers-copy.TXT', sep='|', error_bad_lines=False)
speakers_filtered = speakers[(speakers['SUBSET'] == 'train-clean-100') | (speakers['SUBSET'] == 'train-clean-360')]
speakers_filtered = speakers_filtered.copy()
speakers_filtered['CODE'] = speakers_filtered['NAME'].astype('category').cat.codes

print('read and filtered metdata')

unique_speakers = np.unique(speakers_filtered['CODE'])
for speaker in unique_speakers:
    try:
        os.makedirs(root + '/train-gram/' + str(speaker))
        print('created folder for speaker {}'.format(speaker))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

print('created all directories')

my_dpi = 120

for index, row in speakers_filtered.iterrows():
    dir_ = root + '/' + row['SUBSET'] + '/' + str(row['ID']) + '/'
    print('working on df row {}, spaker {}'.format(index, row['CODE']))
    if not os.path.exists(dir_):
        print('dir {} not exists, skipping'.format(dir_))
        continue

    files_iter = Path(dir_).glob('**/*.flac')
    files_ = [str(f) for f in files_iter]

    for f in files_:
        ay, sr = librosa.load(f)
        duration = ay.shape[0] / sr
        start = 0
        while start + 5 < duration:
            slice_ = ay[start * sr: (start + 5) * sr]
            start = start + 5 - 1
            x = librosa.stft(slice_)
            xdb = librosa.amplitude_to_db(abs(x))
            plt.figure(figsize=(227 / my_dpi, 227 / my_dpi), dpi=my_dpi)
            plt.axis('off')
            librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='log')
            plt.savefig(root + '/train-gram/' + str(row['CODE']) + '/' + uuid.uuid4().hex + '.png', dpi=my_dpi)
            plt.close()

    print('work done on index {}, speaker {}'.format(index, row['CODE']))
