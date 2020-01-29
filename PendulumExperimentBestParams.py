import helperfns
import kerastuner
import networkarch as net
import tensorflow as tf

#tf.keras.backend.set_floatx('float64')

# settings related to dataset
data_name = 'Pendulum'
folder_name = 'exp2_best'
num_data_train_files = 6
num_initial_conditions = 5000  # per training file

# settings related to system
num_steps = 51
input_dim = 2
delta_t = 0.02

hp = kerastuner.HyperParameters()

# settings related to training
epochs = 15
hp.Fixed('learning_rate', 10.0 ** (-3))
batch_size = 128


# settings related to network architecture
hp.Fixed("omega_num_complex_pairs", 1)
hp.Fixed("omega_num_real", 0)

hp.Fixed("enc_dec_num_layers", 2)
hp.Fixed('enc_dec_layer_width', 80)

hp.Fixed("omega_num_layers", 1)
hp.Fixed('omega_layer_width', 170)

# settings related to loss function
num_steps_for_loss = 30
hp.Fixed("reconstruction_loss_lambda_exponent", -3)
hp.Fixed("L_inf_loss_lambda_exponent", -9)
hp.Fixed("enc_dec_l2_exponent", -14)
hp.Fixed("omega_l2_exponent", -14)

data_train_tensor = helperfns.load_training_data(data_name, num_data_train_files, num_steps, num_steps_for_loss)
data_val_tensor = helperfns.load_eval_data(data_name, num_steps, num_steps_for_loss)

hp.Fixed("times_recon_loss_only", int(data_train_tensor.shape[0] / batch_size)) # 1 epoch


deep_koop = net.DeepKoopmanHyperModel(input_dim, num_steps,
    num_steps_for_loss, delta_t)

#tuner = kerastuner.tuners.Hyperband(
#    deep_koop,
#    objective=kerastuner.Objective("val_prediction_loss", direction="min"),
#    max_epochs=epochs,
#    directory=data_name,
#    project_name=folder_name,
#    executions_per_trial=3,
#    seed=42,
#    tune_new_entries=False,
#    hyperparameters=hp,
#)
tuner = kerastuner.tuners.RandomSearch(
    deep_koop,
    objective=kerastuner.Objective("val_prediction_loss", direction="min"),
    max_trials=1,
    directory=data_name,
    project_name=folder_name,
    executions_per_trial=200,
    seed=42,
    tune_new_entries=False,
    hyperparameters=hp,
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_prediction_loss', patience=2, min_lr=1.0e-8)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_prediction_loss', patience=3, restore_best_weights=True)

tuner.search_space_summary()

tuner.search(
    x=data_train_tensor,
    y=data_train_tensor,
    validation_data=(data_val_tensor, data_val_tensor),
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[reduce_lr, stop_early],
)

tuner.results_summary()
