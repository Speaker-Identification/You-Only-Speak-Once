import os
import sys
import logging
import boto3
from decouple import config

import numpy as np
from flask import Flask, render_template, request, Response,jsonify

from .preprocessing import extract_fbanks
from .predictions import get_embeddings, get_cosine_distance

app = Flask(__name__)
s3 = boto3.client('s3')

DATA_DIR = 'data_files/'
THRESHOLD = 0.45    # play with this value. you may get better results

sys.path.append('..')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login/<string:username>', methods=['POST'])
def login(username):

    filename = _save_file(request, username)
    print('filename ', filename, flush=True)

    fbanks = extract_fbanks(filename)
    embeddings = get_embeddings(fbanks)
    stored_embeddings = np.load(DATA_DIR + username + '/embeddings.npy')
    stored_embeddings = stored_embeddings.reshape((1, -1))

    distances = get_cosine_distance(embeddings, stored_embeddings)
    print('mean distances', np.mean(distances), flush=True)
    positives = distances < THRESHOLD
    positives_mean = np.mean(positives)
    print('positives mean: {}'.format(positives_mean), flush=True)
    if positives_mean >= 0.65:
        return jsonify(username=username,result="true")
    else:
        return jsonify(username=username,result="false")


@app.route('/register/<string:username>', methods=['POST'])
def register(username):
    filename = _save_file(request, username)
    _upload_file_to_s3(filename, username)
    fbanks = extract_fbanks(filename)
    embeddings = get_embeddings(fbanks)
    print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
    mean_embeddings = np.mean(embeddings, axis=0)
    np.save(DATA_DIR + username + '/embeddings.npy', mean_embeddings)
    return Response('registered', mimetype='application/json')

def _upload_file_to_s3(filename, username):
    """
    Uploads file to S3 bucket using S3 client object
    :return: None
    """
    bucketname = config('S3_BUCKET')
    objectname = 'audio-samples/'+str(username)+'.wav'
    response = s3.upload_file(filename, bucketname, objectname)
    print(response)

def _save_file(request_, username):
    file = request_.files['file']
    dir_ = DATA_DIR + username
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    filename = DATA_DIR + username + '/sample.wav'
    file.save(filename)

    return filename

