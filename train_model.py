"""
    Script for training demo genre classification in pure tf (no use of dzr_audio)
"""
import os
import argparse
import tensorflow as tf
import numpy as np

from keras_model import build_model, build_model2
from data_pipeline_pandas import get_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.compat.v1 import ConfigProto

save_best_loss = ModelCheckpoint(filepath="Modelsaves/best_loss_model", monitor='loss', save_best_only=True,
                                 save_weights_only=True, verbose=1)
save_best_acc = ModelCheckpoint(filepath="Modelsaves/best_acc_model", monitor='accuracy', save_best_only=True,
                                save_weights_only=True, verbose=1)
save_val_best_loss = ModelCheckpoint(filepath="Modelsaves/best_loss_val_model", monitor='val_loss', save_best_only=True,
                                     save_weights_only=True, verbose=1)
save_val_best_acc = ModelCheckpoint(filepath="Modelsaves/best_acc_val_model", monitor='val_accuracy',
                                    save_best_only=True,
                                    save_weights_only=True, verbose=1)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    batch_size = 16

    model = build_model(batch_size)
    model.summary()

    try:
        model.load_weights("Modelsaves/best_loss_model")
    except Exception as e:
        print(e)

    data = np.load('Data/ProcessedData.npy')
    labels = np.load('Data/Processedlabels.npy')

    val_data = (np.load('Data/ProcessedData_val.npy'), np.load('Data/Processedlabels_val.npy'))

    print(data.shape)
    print(labels.shape)
    # print(val_data.shape)


    history = model.fit(data, labels, batch_size=batch_size, validation_data=val_data, epochs=10, verbose=1,
                        shuffle=True, callbacks=[save_best_loss, save_best_acc, save_val_best_loss, save_val_best_acc])

    np.save('history.npy', history.history)
