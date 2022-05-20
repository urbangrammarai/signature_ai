"""
Tools to work within a
nvcr.io/nvidia/tensorflow:21.11-tf2-py3 Keras container
"""

import os, json, time, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


def flush(folder, subfolders=["json", "logs", "model", "pred"]):
    for f in subfolders:
        shutil.rmtree(folder + f)
        os.mkdir(folder + f)


def fit_phase(
    model,
    train_dataset,
    validation_dataset,
    secret_dataset,
    log_folder=None,
    pred_folder=None,
    model_folder=None,
    json_folder=None,
    specs=None,
    chips_all=None,
    epochs=250,
    early_stopping_delta=0.01,
    patience=3,
    batch_size=32,
    verbose=False,
    **kwargs
):
    """
    Train model with logging, write predicted labels,
    serialise model, and save metadata
    ...

    Arguments
    ---------
    model : keras.Model
            Model to fit
    train_dataset : str
               Path to a folder with train data
    validation_dataset  : str
              Path to a folder with validation data
    secret_dataset  : str
               Path to a folder with secret data
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
    chips_all : None/ndarray/tf.Dataset
            [Optional. Default=None] Array with full set of features to use
            to obtain predicted labels from
    epochs : int
             [Optional. Default=250] Epochs for fitting
    early_stopping_delta : float
        [Optional. Default=0.01]
        Minimum change in the monitored quantity to qualify as an improvement, 
        i.e. an absolute change of less than min_delta, will count as no improvement.
    patience : int
        [Optional. Default=3]
        Number of epochs with no improvement after which training will be stopped.
    batch_sizie : int
        batch size of ImageDataGenerator
    verbose : Boolean
             [Optional. Default=False] If True, print model summary and
             fitting progress

    Returns
    -------
    meta : dict
           Metadata object (which has also been saved as a json file
    """
    if verbose:
        print(model.summary())
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience, min_delta=early_stopping_delta, verbose=verbose)]
    if log_folder is not None:
        callbacks.append(
            TensorBoard(log_dir=os.path.join(log_folder, model.name), histogram_freq=1)
        )
    if model_folder is not None:
        callbacks.append(
            ModelCheckpoint(
                filepath=f"{model_folder}/{model.name}_best",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
            )
        )
    if verbose:
        print(f"creating ImageDataGenerators...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_dataset,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse')
    
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        validation_dataset,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse')
    
    if verbose:
        print(f"training...")
    t0 = time.time()
    h = model.fit(
        train_generator,
        epochs=epochs,
        shuffle=True,
        validation_data=validation_generator,
        verbose=verbose,
        callbacks=callbacks,
        **kwargs
    )
    if "class_weights" in kwargs:
        specs["class_weights"] = kwargs["class_weights"]
    t1 = time.time()
    if specs is not None:
        specs["runtime"] = t1 - t0
        if verbose:
            print(f"time elapsed: {(t1 - t0):9.1f}s")

    if model_folder is not None:
        model.save(f"{model_folder}/{model.name}", save_format="tf")
    if pred_folder:
        specs["pred_folder"] = pred_folder

    if json_folder:
        meta = build_meta_json(
            model,
            specs,
            train_dataset,
            validation_dataset,
            secret_dataset,
            batch_size,
            verbose=True,
        )
        os.makedirs(json_folder, exist_ok=True)
        with open(
            os.path.join(json_folder, model.name + ".json"), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(meta, indent=4, cls=NumpyEncoder).replace("NaN", "null"))
    return h


def build_meta_json(
    model, specs, train_dataset, validation_dataset, secret_dataset, batch_size, verbose=False
):
    """
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
    train_dataset : tensorflow.Dataset
               Tensorflow Dataset with (chips, labels) to fit on training
               stage
    validation_dataset  : tensorflow.Dataset
               Tensorflow Dataset with (chips, labels) for validation
    secret_dataset  : tensorflow.Dataset
               Tensorflow Dataset with (chips, labels) for final validation

    Returns
    -------
    meta : dict
           Metadata object (which has also been saved as a json file
    """
    nn, bridge, toplayer, n_class = model.name.split("_")

    meta = {
        # Metadata about run
        "meta_n_class": n_class,
        "meta_class_map": specs["meta_class_map"],
        "meta_class_names": specs["meta_class_names"],
        "meta_chip_size": specs["meta_chip_size"],
        # Model
        "model_name": model.name,
        "model_bridge": bridge,
        "model_toplayer": toplayer,
    }

    if "runtime" in specs:
        meta["meta_runtime"] = specs["runtime"]
    
    if verbose:
        print(f"creating ImageDataGenerators")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_dataset,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False,
    )
    
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        validation_dataset,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )
    secret_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    secret_generator = validation_datagen.flow_from_directory(
        secret_dataset,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    subsets = {
        "train": train_generator,
        "val": validation_generator,
        "secret": secret_generator,
    }
    # Performance
    for subset in subsets:
        if verbose:
            print(f"assessing performance of {subset} dataset")
        dataset = subsets[subset]
        y_pred_probs = model.predict(dataset)
        if "pred_folder" in specs:
            os.makedirs(specs['pred_folder'], exist_ok=True)
            np.save(os.path.join(specs["pred_folder"], model.name + f"_{subset}.npy"), y_pred_probs)
            if verbose:
                print(f"prediction of {subset} saved")
        y_pred = np.argmax(y_pred_probs, axis=1)
        y = dataset.labels
        top_prob, wc_accuracy, wc_top_prob = within_class_metrics(
            y, y_pred, y_pred_probs
        )
        # Accuracy
        meta[f"perf_model_accuracy_{subset}"] = accuracy(y, y_pred)
        # Prob for top class
        meta[f"perf_avg_prob_top_{subset}"] = top_prob
        # Within-class accuracy
        meta[f"perf_within_class_accuracy_{subset}"] = wc_accuracy
        # Within-class avg. prob for top class
        meta[f"perf_within_class_avg_prob_top_{subset}"] = wc_top_prob
        # Full confusion matrix
        meta[f"perf_confusion_{subset}"] = confusion_matrix(y, y_pred, int(n_class))
        if verbose:
            print(
                f"perf_model_accuracy for {subset}: {meta[f'perf_model_accuracy_{subset}']}"
            )

    return meta


def accuracy(y, y_pred):
    a = tf.keras.metrics.Accuracy()
    a.update_state(y, y_pred)
    return a.result().numpy()


def within_class_metrics(y, y_pred, y_probs):
    top_prob = np.zeros(y_pred.shape)
    wc_accuracy = np.zeros(y_probs.shape[1]).tolist()
    wc_top_prob = np.zeros(y_probs.shape[1]).tolist()
    for c in range(y_probs.shape[1]):
        c_id = y == c
        # Top prob
        top_prob[c_id] = y_probs[c_id, c]
        # WC accuracy
        wc_accuracy[c] = accuracy(y[c_id], y_pred[c_id])
        # WC top prob
        wc_top_prob[c] = y_probs[c_id, c].mean()
    top_prob = top_prob.mean()
    return top_prob, wc_accuracy, wc_top_prob


def confusion_matrix(y, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    pairs = np.vstack((y, y_pred)).T
    for c1 in range(n_classes):
        for c2 in range(n_classes):
            cm[c1, c2] = ((pairs[:, 0] == c1) * (pairs[:, 1] == c2)).sum()
    return cm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)