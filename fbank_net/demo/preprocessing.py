import librosa
import numpy as np
import python_speech_features as psf


def get_fbanks(audio_file):
    
    def normalize_frames(signal, epsilon=1e-12):
        return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])

    y, sr = librosa.load(audio_file, sr=16000)
    assert sr == 16000

    trim_len = int(0.25 * sr)
    if y.shape[0] < 1 * sr:
        # if less than 1 seconds, don't use that audio
        return None

    y = y[trim_len:-trim_len]

    # frame width of 25 ms with a stride of 15 ms. This will have an overlap of 10s
    filter_banks, energies = psf.fbank(y, samplerate=sr, nfilt=64, winlen=0.025, winstep=0.01)
    filter_banks = normalize_frames(signal=filter_banks)

    filter_banks = filter_banks.reshape((filter_banks.shape[0], 64, 1))
    return filter_banks


def extract_fbanks(path):
    fbanks = get_fbanks(path)
    num_frames = fbanks.shape[0]

    # sample sets of 64 frames each

    numpy_arrays = []
    start = 0
    while start < num_frames + 64:
        slice_ = fbanks[start:start + 64]
        if slice_ is not None and slice_.shape[0] == 64:
            assert slice_.shape[0] == 64
            assert slice_.shape[1] == 64
            assert slice_.shape[2] == 1

            slice_ = np.moveaxis(slice_, 2, 0)
            slice_ = slice_.reshape((1, 1, 64, 64))
            numpy_arrays.append(slice_)
        start = start + 64

    print('num samples extracted: {}'.format(len(numpy_arrays)))
    return np.concatenate(numpy_arrays, axis=0)
