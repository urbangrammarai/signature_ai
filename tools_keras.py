"""
Tools to work within a
nvcr.io/nvidia/tensorflow:21.03-tf2-py3 Keras container
"""

import os, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def flush():
    return None

def fit_phase(
        model,
        xy_train,
        xy_val,
        log_folder=None,
        pred_folder=None,
        model_folder=None,
        json_folder=None,
        specs=None,
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
    xy_val  : tuple
              Pair of arrays with features and labels to fit on validation
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
    specs : dict
            [Optional. Default=None] Specs about the run to store as JSON with
            performance
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
    x_train, y_train = xy_train
    x_val, y_val = xy_val
    callbacks = None
    if log_folder is not None:
        callbacks = [TensorBoard(
            log_dir=os.path.join(log_folder, model.name),
            histogram_freq=1
        )]
    t0 = time.time()
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
    t1 = time.time()
    if specs is not None:
        specs['runtime'] = t1 - t0
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

def build_meta_json(model, specs, xy_train, xy_val, xy_secret, xy_all):
    '''
    Compile metadata about model, training specs, and performance
    metrics
    ...

    Arguments
    ---------
    model : keras.Model
            Fitted model
    specs : dict
            Specs of the model being fit. Requires the following keys:
                - `meta_class_map`: mapping of aggregated classes
                - `meta_class_names`: class names
                - `meta_chip_size`: chip size, expressed in pixels
    xy_train : tuple
               Pair of arrays with features and labels to fit on training
               stage
    xy_val  : tuple
              Pair of arrays with features and labels for validation
    xy_secret  : tuple
                 Pair of arrays with features and labels for final validation
    xy_all  : tuple
              Pair of arrays with all features and labels in order for final
              predictions

    Returns
    -------
    meta : dict
           Metadata object (which has also been saved as a json file
    '''
    nn, bridge, toplayer, n_class = model.name.split('_')
    meta = {
        # Metadata about run
        'meta_n_class': n_class,
        'meta_class_map': specs['meta_class_map'],
        'meta_class_names': specs['meta_class_names'],
        'meta_chip_size': specs['meta_chip_size'],
        'meta_runtime': specs['runtime'], 
        # Model
        'model_name': model.name,
        'model_bridge': bridge,
        'model_toplayer': toplayer,
    }
    subsets = {
        'train': xy_train, 'val': xy_val, 'secret': xy_secret, 'all': xy_all
    }
    # Performance
    for subset in ['train', 'val', 'secret', 'all']:
        x, y = subsets[subset]
        y_pred_probs = model.predict(x)
        y_pred = np.argmax(y_pred_probs, axis=1)
        # Accuracy
        meta[f'perf_model_accuracy_{subset}'] = accuracy(y, y_pred)
        # Prob for top class
        meta[f'perf_avg_prob_top_{subset}'] = None
        # Within-class accuracy
        meta[f'perf_within_class_accuracy_{subset}'] = None
        # Within-class avg. prob for top class
        meta[f'perf_within_class_avg_prob_top_{subset}'] = None
        # Full confusion matrix
        meta[f'perf_confusion_{subset}'] = None
    return meta

def accuracy(y, y_pred):
    a = tf.keras.metrics.Accuracy()
    a.update_state(y, y_pred)
    return a.result().numpy()

