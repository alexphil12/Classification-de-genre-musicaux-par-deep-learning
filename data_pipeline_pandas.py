import json
import time
import os
from os.path import join
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from utils_1 import one_hot_label, read, Spectre_de_mel, Spectrogramme_de_mel

DATASET_DIR = "fma_small/"


def get_dataset(input_csv, n_tracks, n_sample):
    # build dataset from csv file
    # n_sample=11025
    datafiles = np.array([])
    labels = np.array([])
    #data = np.empty(n_sample)
    NFFT = 2048
    Banque = Spectre_de_mel(20000, 44100, 40, NFFT)

    Q = n_sample // NFFT
    Sortie = np.empty((n_tracks, Q, NFFT // 2))

    # Données de test
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i < n_tracks + 1:
                datafiles = np.append(datafiles, row[0])
                labels = np.append(labels, row[1])
            else:
                break
            i += 1
    datafiles = np.delete(datafiles, 0)
    labels = np.delete(labels, 0)

    ## Données de validation
    # with open(input_csv, newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     i = 0
    #     for row in reader:
    #         if 3000  < i < 3000 + n_tracks + 1:
    #             datafiles = np.append(datafiles, row[0])
    #             labels = np.append(labels, row[1])
    #             print(i)
    #         elif i == 3000 + n_tracks + 1:
    #             break
    #         i += 1


    onehotlabels = one_hot_label(labels, n_tracks)
    print(set(labels))

    for i in range(n_tracks):
        file_directory = ''.join([DATASET_DIR, datafiles[i]])
        data = read(file_directory, n_sample)
        print(i)
        Sortie[i] = Spectrogramme_de_mel(Banque, data)

    Sortie = np.expand_dims(Sortie, axis=3)

    print(Sortie.shape)
    return Sortie, onehotlabels


# test dataset data generation
if __name__ == "__main__":
    data, labels = get_dataset("fma_small.csv", 200, 200000)
    np.save('ProcessedData_val.npy', data)
    np.save('Processedlabels_val.npy', labels)
