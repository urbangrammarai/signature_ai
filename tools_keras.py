"""
Tools to work within a
nvcr.io/nvidia/tensorflow:21.03-tf2-py3 Keras container
"""

import os, json
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

def flush():
    return None

def fit_phase(
        model,
        xy_train,
        xy_test,
        log_folder=None,
        pred_folder=None,
        model_folder=None,
        json_folder=None,
        x_all=None,
        epochs=250,
        batch_size=512,
        verbose=False
):
    '''
    Train model with logging, write predicted labels,
    serialise model, and save metadata
    ...
    
    Arguments
    ---------
    model : keras.Model
            Model to fit
    xy_train : tuple
               Pair of arrays with features and labels to fit on training
               stage
    xy_test : tuple
              Pair of arrays with features and labels to fit on training
              stage
    log_folder : None/str
                 [Optional. Default=None] Path to folder to store log files.
                 A new subfolder will be created with the model name to store
                 logs
    pred_folder : None/str
                  [Optional. Default=None] Path to folder to store predicted
                  labels. The file with the labels will be named after the
                  `model.name`
    model_folder : None/str
                   [Optional. Default=None] Path to folder to store serialised
                   version of the model. A subfolder will be created with the
                   `model.name` attribute to store all components
    json_folder : None/str
                  [Optional. Default=None] Path to folder to store metadata
                  JSON file, named after `model.name`
    x_all : None/ndarray
            [Optional. Default=None] Array with full set of features to use
            to obtain predicted labels from
    epochs : int
             [Optional. Default=250] Epochs for fitting
    batch_size : int
             [Optional. Default=512] Batch size
    verbose : Boolean
             [Optional. Default=False] If True, print model summary and
             fitting progress

    Returns
    -------
    meta : dict
           Metadata object (which has also been saved as a json file
    '''
    if verbose:
        print(model.summary())
    xy_train = x_train, y_train
    xy_test = x_test, y_test
    callbacks = None
    if log_folder is not None:
        callbacks = [TensorBoard(
            log_dir=os.path.join(log_folder, model.name),
            histogram_freq=1
        )]
    h = model.fit(
        x_train, 
        y_train,
        epochs=epochs,
        batch_size=epochs,
        shuffle=True,
        validation_data=(x_test, y_test),
        verbose=verbose,
        callbacks=callbacks,
    )
    if x_all is None:
        x_all = np.vstack((x_train, x_test))
    if pred_folder:
        y_pred = model.predict(x_all)
        np.save(
            os.path.join(pred_folder, model.name+'.npy'), y_pred
        )
    if model_folder:
        model.save(os.path.join(model_folder, model.name))
    if json_folder:
        meta = build_meta_json(model)
        with open(
            os.path.join(json_folder, model.name+'.json'), 
            'w',
            encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=4)
    return h

def build_meta_json():
    '''
    Compile metadata about model, training specs, and performance
    metrics
    ...

    Arguments
    ---------

    Returns
    -------
    meta : dict
           Metadata object (which has also been saved as a json file
    '''
    return meta



