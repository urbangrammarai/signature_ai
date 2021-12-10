import numpy

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet_v2, vgg19, efficientnet


def model_builder(
    model_name, 
    bridge, 
    top_layer_neurons, 
    n_labels, 
    input_shape=(32, 32, 3), 
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"],
):
    """Return a Keras model according to specification

    Assumes chips with values [0, 255]

    Parameters
    ----------
    model_name : str
        {"resnet50", "vgg19", "efficientnet"}
    bridge : str
        {"pooling", "flatten"}
    top_layer_neurons : int
        number of neurons of `layers.Dense(n, activation='relu')`
    n_labels : int
        number of labels
    input_shape : tuple, default (32, 32, 3)
        shape of an input chip
    optimizer : str, keras optimizer, default "adam"
        model optimizer
    loss : str, keras loss, default "sparse_categorical_crossentropy"
        loss function
    metrics : list, default ["accuracy"]
        list of metrics to capture

    Returns
    -------
    keras.Model
        compiled model

    """
    base_models = {
        "resnet50": keras.applications.ResNet50,
        "vgg19": keras.applications.VGG19,
        "efficientnet": keras.applications.EfficientNetB4,
    }
    bridges = {"pooling": layers.GlobalAveragePooling2D(), "flatten": layers.Flatten()}
    
    preprocessing = {
        "resnet50": resnet_v2.preprocess_input,
        "vgg19": vgg19.preprocess_input,
        "efficientnet": efficientnet.preprocess_input,
    }

    # initialise base model
    base = base_models[model_name](
            weights="imagenet",
            input_shape=(224, 224, 3),
            include_top=False,
        )
    # freeze base model
    base.trainable = False
    
    # create input
    inputs = keras.Input(shape=input_shape)
    # resize
    x = layers.Resizing(224, 224, crop_to_aspect_ratio=True)(inputs)
    # preprocess using model preprocessing
    x = preprocessing[model_name](x)
    # add base
    x = base(x, training=False)
    # add bridge
    x = bridges[bridge](x)
     # add Dense relu layer
    x = layers.Dense(top_layer_neurons, activation="relu")(x)
    # add softmax classfier
    predictions = layers.Dense(n_labels, activation="softmax")(x)
    
    model = keras.Model(
            inputs,
            predictions,
            name=f"{model_name}_{bridge}_{top_layer_neurons}_{n_labels}"
    )
    
    # compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model


def relabel(labels):
    """Encode an input array into integers
    
    Parameters
    ----------
    labels : array
    
    Returns
    -------
    array
    
    Examples
    --------
    >>> arr = numpy.array(["a", "b", "a", "c"])
    >>> relabel(arr)
    array([0, 1, 0, 2])
    """
    unique = numpy.unique(labels)
    out = numpy.zeros(shape=labels.shape, dtype=int)
    for i, label in enumerate(unique):
        out[labels == label] = i
        
    return out


def class_merger(labels, group_mapping):
    """Merge classes together
    
    Parameters
    ----------
    labels : array
        input labels to be remapped
    group_mapping : list of lists
        mapping of classes to groups. One list encodes one group.
        
    Returns
    -------
    array
        array of ints 0..n
    
    Example
    -------
    >>> arr = numpy.array(["a", "b", "a", "c", "d"])
    >>> groups = [
    ...    ["a", "c"],
    ...    ["b", "d"],
    ...]
    >>> class_merger(arr, groups)
    array([0, 1, 0, 0, 1])
    
    """
    out = numpy.zeros(shape=labels.shape, dtype=int)
    for i, group in enumerate(group_mapping):
        out[numpy.isin(labels, group_mapping[i])] = i
        
    return out


def balancer(labels, max_ratio=2, verbose=True):
    """
    Generate mask removing labels and chips of excessive classes
    
    Chips and labels are assumed to be shuffled, i.e. their order has
    no meaning in space.
    
    Parameters
    ----------
    labels : array
        array of int labels (labels must be 0..n)
    max_ratio : float, default 2
        maximum ratio of the smallest to the largest class. 
        If max_ratio == 2 and the least abundant class has 10 labels,
        no class will have more than 20 labels.
    verbose : bool, default True
        print summary of the balanced result
        
    Returns
    -------
    array
        boolean array to be used as a mask on both labels and chips
    """
    unique, counts = numpy.unique(labels, return_counts=True)
    maximum = max_ratio * counts.min()
    
    mask = numpy.zeros(shape=labels.shape, dtype=bool)

    counts = {v:0 for v in unique}
    for i, v in enumerate(labels.flatten()):
        if counts[v] < maximum:
            counts[v] += 1
            mask[i] = True
    
    if verbose:
        print(f"Total number of selected chips: {mask.sum()} out of {mask.shape[0]}\nCounts:\n", counts)
            
    return mask.flatten()
