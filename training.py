import os
import time

import numpy as np
import tensorflow as tf

import helperfns
import networkarch as net

tf.keras.backend.set_floatx('float64')


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
    deep_koop = net.DeepKoopman(params)

    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    #x = tf.compat.v1.placeholder(tf.float64, [max_shifts_to_stack + 1, None, params['widths'][0]])
    #y = deep_koop(x,)


    # DEFINE LOSS FUNCTION
    #trainable_var = tf.compat.v1.trainable_variables()
    #loss1, loss2, loss3, loss_Linf, loss = deep_koop.recon_loss(x, y), deep_koop.prediction_loss(x, y), deep_koop.linearity_loss(x, y), deep_koop.linf_loss(x, y), deep_koop.total_loss(x, y),

    # CHOOSE OPTIMIZATION ALGORITHM
    #optimizer = helperfns.choose_optimizer(params, loss, trainable_var)
    #optimizer_autoencoder = helperfns.choose_optimizer(params, loss1, trainable_var)

    deep_koop.compile(
        optimizer=tf.optimizers.Adam(params['learning_rate']),
        loss=deep_koop.total_loss,
        metrics=[deep_koop.recon_loss, deep_koop.prediction_loss, deep_koop.linearity_loss, deep_koop.linf_loss],
        run_eagerly=True,
    )

    # Load training data
    for f in range(params['data_train_len']):
        file_num = f + 1  # 1...data_train_len
        data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], file_num)), delimiter=',',
                                    dtype=np.float64)
        data_train_tensor_temp = helperfns.stack_data(data_train, max_shifts_to_stack, params['len_time'])
        if f==0: data_train_tensor = data_train_tensor_temp
        else: data_train_tensor = tf.concat([data_train_tensor, data_train_tensor_temp], axis=1)
    data_train_tensor = tf.transpose(data_train_tensor, perm=[1, 0, 2])

    # Load validation data
    # data is num_steps x num_examples x n but load flattened version (matrix instead of tensor)
    data_val = np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)
    data_val_tensor = helperfns.stack_data(data_val, max_shifts_to_stack, params['len_time'])
    data_val_tensor = tf.transpose(data_val_tensor, perm=[1, 0, 2])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    deep_koop.fit(
        x=data_train_tensor,
        y=data_train_tensor,
        validation_data=(data_val_tensor, data_val_tensor),
        shuffle=True,
        epochs=params['num_passes_per_file'],
        batch_size=params["batch_size"],
        callbacks=[reduce_lr, stop_early],
    )


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
