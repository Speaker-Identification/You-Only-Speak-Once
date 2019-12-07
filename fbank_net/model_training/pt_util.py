import glob
import os
import pickle

import torch


def _remove_files(files):
    for f in files:
        return os.remove(f)


def assert_dir_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model, epoch, out_path):
    assert_dir_exits(out_path)
    chk_files = glob.glob(out_path + '*.pth')
    _remove_files(chk_files)
    torch.save(model.state_dict(), out_path + str(epoch) + '.pth')
    print('model saved for epoch: {}'.format(epoch))


def save_objects(obj, epoch, out_path):
    assert_dir_exits(out_path)
    dat_files = glob.glob(out_path + '*.dat')
    _remove_files(dat_files)
    # object should be tuple
    with open(out_path + str(epoch) + '.dat', 'wb') as output:
        pickle.dump(obj, output)

    print('objects saved for epoch: {}'.format(epoch))


def restore_model(model, out_path):
    chk_file = glob.glob(out_path + '*.pth')

    if chk_file:
        chk_file = str(chk_file[0])
        print('found modeL {}, restoring'.format(chk_file))
        model.load_state_dict(torch.load(chk_file))
    else:
        print('Model not found, using untrained model')
    return model


def restore_objects(out_path, default):
    data_file = glob.glob(out_path + '*.dat')
    if data_file:
        data_file = str(data_file[0])
        print('found data {}, restoring'.format(data_file))
        with open(data_file, 'rb') as input_:
            obj = pickle.load(input_)

        return obj
    else:
        return default
