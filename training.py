import os
import time

import numpy as np
import tensorflow as tf

import helperfns
import networkarch as net

#tf.compat.v1.disable_eager_execution()
tf.keras.backend.set_floatx('float64')


def try_net(data_val, params):
    """Run a random experiment for particular params and data.

    Arguments:
        data_val -- array containing validation dataset
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Changes params dict
        Saves files
        Builds TensorFlow graph (reset in main_exp)
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

    data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], 1)), delimiter=',',
                    dtype=np.float64)
    data_train_tensor = helperfns.stack_data(data_train, max_shifts_to_stack, params['len_time'])
    data_train_tensor = tf.transpose(data_train_tensor, perm=[1, 0, 2])

    data_val_tensor = helperfns.stack_data(data_val, max_shifts_to_stack, params['len_time'])

    deep_koop.fit(
        x=data_train_tensor,
        y=data_train_tensor,
        epochs=50,
        batch_size=params["batch_size"],
    )

    # LAUNCH GRAPH AND INITIALIZE
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    csv_path = params['model_path'].replace('model', 'error')
    csv_path = csv_path.replace('ckpt', 'csv')
    print(csv_path)

    num_saved_per_file_pass = params['num_steps_per_file_pass'] / 20 + 1
    num_saved = np.floor(num_saved_per_file_pass * params['data_train_len'] * params['num_passes_per_file']).astype(int)
    train_val_error = np.zeros([num_saved, 16])
    count = 0
    best_error = 10000

    data_val_tensor = helperfns.stack_data(data_val, max_shifts_to_stack, params['len_time'])

    start = time.time()
    finished = 0
    saver.save(sess, params['model_path'])

    # TRAINING
    # loop over training data files
    for f in range(params['data_train_len'] * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % params['data_train_len']) + 1  # 1...data_train_len

        if (params['data_train_len'] > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], file_num)), delimiter=',',
                                    dtype=np.float64)
            data_train_tensor = helperfns.stack_data(data_train, max_shifts_to_stack, params['len_time'])
            num_examples = data_train_tensor.shape[1]
            num_batches = int(np.floor(num_examples / params['batch_size']))

        ind = np.arange(num_examples)
        np.random.shuffle(ind)
        data_train_tensor = data_train_tensor[:, ind, :]

        # loop over batches in this file
        for step in range(params['num_steps_per_batch'] * num_batches):

            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0

            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]

            feed_dict_train = {x: batch_data_train}
            feed_dict_train_loss = {x: batch_data_train}
            feed_dict_val = {x: data_val_tensor}

            if (not params['been5min']) and params['auto_first']:
                sess.run(optimizer_autoencoder, feed_dict=feed_dict_train)
            else:
                sess.run(optimizer, feed_dict=feed_dict_train)

            if step % 20 == 0:
                train_error = sess.run(loss, feed_dict=feed_dict_train_loss)
                val_error = sess.run(loss, feed_dict=feed_dict_val)
                train_error, val_error = train_error.mean(), val_error.mean()

                if val_error < (best_error - best_error * (10 ** (-5))):
                    best_error = val_error.copy()
                    saver.save(sess, params['model_path'])
                    print("New best val error %f" % (best_error))

                train_val_error[count, 0] = train_error
                train_val_error[count, 1] = val_error
                train_val_error[count, 4] = sess.run(loss1, feed_dict=feed_dict_train_loss).mean()
                train_val_error[count, 5] = sess.run(loss1, feed_dict=feed_dict_val).mean()
                train_val_error[count, 6] = sess.run(loss2, feed_dict=feed_dict_train_loss).mean()
                train_val_error[count, 7] = sess.run(loss2, feed_dict=feed_dict_val).mean()
                train_val_error[count, 8] = sess.run(loss3, feed_dict=feed_dict_train_loss).mean()
                train_val_error[count, 9] = sess.run(loss3, feed_dict=feed_dict_val).mean()
                train_val_error[count, 10] = sess.run(loss_Linf, feed_dict=feed_dict_train_loss).mean()
                train_val_error[count, 11] = sess.run(loss_Linf, feed_dict=feed_dict_val).mean()
                if np.isnan(train_val_error[count, 10]):
                    params['stop_condition'] = 'loss_Linf is nan'
                    finished = 1
                    break

                np.savetxt(csv_path, train_val_error, delimiter=',')
                finished, save_now = helperfns.check_progress(start, best_error, params)
                count = count + 1
                if save_now:
                    train_val_error_trunc = train_val_error[range(count), :]
                    helperfns.save_files(sess, csv_path, train_val_error_trunc, params, weights, biases)
                if finished:
                    break

            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

    # SAVE RESULTS
    train_val_error = train_val_error[range(count), :]
    print(train_val_error)
    params['time_exp'] = time.time() - start
    saver.restore(sess, params['model_path'])
    helperfns.save_files(sess, csv_path, train_val_error, params, weights, biases)
    tf.compat.v1.reset_default_graph()


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
    # data is num_steps x num_examples x n but load flattened version (matrix instead of tensor)
    data_val = np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)
    try_net(data_val, params)
