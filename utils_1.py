"""
    Utility functions for data processing
"""
import tensorflow as tf
import pandas as pd
import pydub
import numpy as np


def read(file, n_samples):
    a = pydub.AudioSegment.from_mp3(file)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y[:].reshape((-1, 2))
        y = y[:, 0]
    del a
    return np.float32(y[200000:200000+n_samples])/2**15

def one_hot_label(labels, n_tracks):
    labdict = {'Pop': [0, 0, 0, 0, 0, 0, 0, 1], 'International': [0, 0, 0, 0, 0, 0, 1, 0],
               'Folk': [0, 0, 0, 0, 0, 1, 0, 0, ], 'Hip-Hop': [0, 0, 0, 0, 1, 0, 0, 0],
               'Electronic': [0, 0, 0, 1, 0, 0, 0, 0], 'Instrumental': [0, 0, 1, 0, 0, 0, 0, 0],
               'Experimental': [0, 1, 0, 0, 0, 0, 0, 0], 'Rock': [1, 0, 0, 0, 0, 0, 0, 0]}
    ohlabels = np.empty((n_tracks, 8))

    for lab, onehot in labdict.items():
        for i in range(len(labels)):
            if labels[i] == lab:
                ohlabels[i] = onehot
    return ohlabels


def triangle_gene(minw, center, maxw, n):
    triangle = square = [0 for i in range(n)]
    square[minw + 1:center + 1] = [1 / (center - minw) for i in range(minw, center)]
    square[center + 1:maxw + 1] = [-1 / (maxw - center) for i in range(center, maxw)]
    for k in range(maxw):
        triangle[k + 1] = triangle[k] + square[k + 1]

    return triangle


def Spectre_de_mel(F_max, F_ech, N_filtre, NFFT):
    freq_basse_mel = 0
    freq_haute_mel = (2595 * np.log10(1 + F_max / 700))
    mel_points = np.linspace(freq_basse_mel, freq_haute_mel, N_filtre + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    N1 = NFFT
    Banque = np.zeros((N_filtre, N1))
    bins = np.floor(N1  * hz_points / F_max)
    for m in range(1, N_filtre - 1):
        f_m_minus = int(bins[m - 1])
        f_m = int(bins[m])
        f_m_plus = int(bins[m + 1])
        for k in range(f_m_minus, f_m):
            Banque[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            Banque[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    return Banque


def Spectrogramme_de_mel(Banque, Signal):
    N = np.size(Banque, axis=1)
    Q = len(Signal) // (N)
    Frame = np.zeros((Q, N))
    Sortie = np.zeros((Q, N))
    Spectr = np.empty((Q, N // 2))
    for i in range(0, Q):
        sig_inter = Signal[i * N:(i + 1) * N]
        Spectre_inter = np.absolute(np.fft.fft(sig_inter))
        Frame[i, :] = Spectre_inter[0:N]
        Frame[i, N // 2:N] = 0
        for k in range(len(Banque)):
            Sortie[i] = Sortie[i] + np.multiply(Banque[k], Frame[i])

    Spectr = Sortie[:, :N // 2]

    return Spectr
