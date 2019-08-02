
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from scipy import signal
from hparam import hparam as hp
import torch
import torch.autograd as grad
import torch.nn.functional as F


def vad(origin):
    origin = np.abs(origin)
    origin /= np.max(origin)

    origin = signal.medfilt(origin, 81)
    gmm = GaussianMixture(
        n_components=3, 
        max_iter=200, 
        covariance_type='diag'
    )
    gmm.fit(np.reshape(origin, (-1, 1)))

    gmm.means_[np.argmin(gmm.means_)] = 0
    gmm.means_[np.argmax(gmm.means_)] = 0
    threshold = np.max(gmm.means_)
    
    length = len(origin)
    idx = 0
    start = 0
    while idx < length:
        if origin[idx] < threshold:
            start = idx
            idy = idx + hp.data.sr // 1000
            while idy < length:
                if origin[idy] < threshold:
                    idy += 1
                    if idy >= length:
                        origin[start:idy] = float('inf')
                        break
                else:
                    idx = idy
                    if idy - start >= hp.data.sr / 10:
                        origin[start:idy] = float('inf')
                    break
            if idy >= length:
                break
        idx += 1

    origin = origin[origin!=float('inf')]
    return origin


def get_centroids(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids


def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid


def get_cossim(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim


def calc_loss(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()    
    return loss, per_embedding_loss