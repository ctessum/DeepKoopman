import os
import time

import numpy as np
import tensorflow as tf
import kerastuner

import helperfns
import networkarch as net

#tf.keras.backend.set_floatx('float64')


def try_net(params):
    """Run a random experiment for particular params and data.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Changes params dict
        Saves files
    """
    # SET UP NETWORK
    deep_koop = net.DeepKoopmanHyperModel(params['input_dim'], params['len_time'],
        params['num_shifts'], params['delta_t'])

    tuner = kerastuner.tuners.Hyperband(
        deep_koop,
        objective=kerastuner.Objective("val_prediction_loss", direction="min"),
        max_epochs=params['num_passes_per_file'],
        directory=params['data_name'],
        project_name=params['folder_name'],
        executions_per_trial=3,
        seed=42,
    )

    data_train_tensor = helperfns.load_training_data(params['data_name'], params['data_train_len'], params['len_time'], params['num_shifts'])
    data_val_tensor = helperfns.load_eval_data(params['data_name'], params['len_time'], params['num_shifts'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_prediction_loss', patience=10)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_prediction_loss', patience=15, restore_best_weights=True)

    tuner.search_space_summary()

    tuner.search(
        x=data_train_tensor,
        y=data_train_tensor,
        validation_data=(data_val_tensor, data_val_tensor),
        shuffle=True,
        epochs=params['num_passes_per_file'],
        batch_size=params["batch_size"],
        callbacks=[reduce_lr, stop_early],
    )

    tuner.results_summary()


def main_exp(params):
    """Set up and run one random experiment.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Changes params dict
        If doesn't already exist, creates folder params['folder_name']
        Saves files in that folder
    """
    helperfns.set_defaults(params)

    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'])

    tf.compat.v1.set_random_seed(params['seed'])
    np.random.seed(params['seed'])
    try_net(params)
