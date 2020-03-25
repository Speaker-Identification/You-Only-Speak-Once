import os
import sys
import logging

import numpy as np
from flask import Flask, render_template, request, Response

from .preprocessing import extract_fbanks
from .predictions import get_embeddings, get_cosine_distance

app = Flask(__name__)

DATA_DIR = 'data_files/'
THRESHOLD = 0.45    # play with this value. you may get better results

sys.path.append('..')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login/<string:username>', methods=['POST'])
def login(username):

    filename = _save_file(request, username)
    fbanks = extract_fbanks(filename)
    embeddings = get_embeddings(fbanks)
    stored_embeddings = np.load(DATA_DIR + username + '/embeddings.npy')
    stored_embeddings = stored_embeddings.reshape((1, -1))

    distances = get_cosine_distance(embeddings, stored_embeddings)
    print('mean distances', np.mean(distances), flush=True)
    positives = distances < THRESHOLD
    positives_mean = np.mean(positives)
    print('positives mean: {}'.format(positives_mean), flush=True)
    if positives_mean >= .65:
        return Response('SUCCESS', mimetype='application/json')
    else:
        return Response('FAILURE', mimetype='application/json')


@app.route('/register/<string:username>', methods=['POST'])
def register(username):
    filename = _save_file(request, username)
    fbanks = extract_fbanks(filename)
    embeddings = get_embeddings(fbanks)
    print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
    mean_embeddings = np.mean(embeddings, axis=0)
    np.save(DATA_DIR + username + '/embeddings.npy', mean_embeddings)
    return Response('', mimetype='application/json')


def _save_file(request_, username):
    file = request_.files['file']
    dir_ = DATA_DIR + username
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    filename = DATA_DIR + username + '/sample.wav'
    file.save(filename)
    return filename
