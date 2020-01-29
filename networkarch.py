import numpy as np
import tensorflow as tf
import kerastuner

import helperfns

def mlp(widths, act_type="relu", reg_type="l2",
    l2=0.01, dropout=None, norm_type="none", norm_before_after=None,
    name="mlp"):
    """ Create a Keras multi-layer perceptron (MLP) model.

    Arguments:
        widths: A list of widths of each layer.
        act_type: Activation type
        reg_type: Regularization type: ["l2", "dropout", "both"]
        l2: l2 regularization strength
        dropout: dropout fraction
        norm_type: type of normalization used: ["none", "batch_norm", "layer_norm"]
        norm_before_after: Whether normalization should occur before or after activation: ["before", "after"]
        name: Name of model

    Returns:
        a Keras model

    Side effects:
        None
    """
    model = tf.keras.Sequential(name=name)

    norm = {"none": None, "batch_norm": tf.keras.layers.BatchNormalization, "layer_norm": tf.keras.layers.LayerNormalization}[norm_type]

    for i in np.arange(len(widths)):
        model.add(tf.keras.layers.Dense(
            widths[i],
            name="%s_%d"%(name,i),
            kernel_regularizer=tf.keras.regularizers.l2(l=l2) if reg_type in ["l2", "both"] else None,
        ))
        if i != len(widths)-1: # Apply last layer without any nonlinearity
            if norm_before_after == "before": # Norm before activation
                if norm!=None: model.add(norm())
                model.add(tf.keras.layers.Activation(act_type))
            else: # Norm after activation
                model.add(tf.keras.layers.Activation(act_type))
                if norm!=None: model.add(norm())
        if reg_type in ["dropout", "both"]: # Add dropout after activation.
            model.add(tf.keras.layers.Dropout(dropout))

    return model

def mlp_hyperpams(hp, prefix):
    """Choose the hyperparameters for an MLP network

    Arguments:
        hp -- Kerastuner HyperParameters object
        prefix -- Prefix for hyperparameter names

    Returns:
        See input arguments for MLP function.

    Side effects:
        None
    """
    num_layers = hp.Int("%s_num_layers"%prefix, 1, 6, default=2)
    layer_width = hp.Choice("%s_layer_width"%(prefix),
        [16, 32, 64, 128, 256, 512, 1028], default=32)
    hidden_layers = [layer_width] * num_layers

    act_type = hp.Choice("%s_activation"%prefix, ("relu", "leaky_relu", "swish", "linear"), default="relu")
    activation = {"relu": tf.nn.relu, "leaky_relu": tf.nn.leaky_relu,
        "swish": tf.nn.swish, "linear":"linear"}[act_type]

    reg_type = hp.Choice("%s_regularization"%prefix, ["none", "dropout", "l2", "both"], default="l2")

    l2 = 10.0 ** hp.Int("%s_l2_exponent"%prefix, -15, 0, default=-2,
        parent_name="%s_regularization"%(prefix), parent_values=["l2", "both"])

    dropout = hp.Float("%s_dropout"%prefix, 0.1, 0.9, step=0.1, default=0.5,
        parent_name="%s_regularization"%prefix, parent_values=["dropout", "both"])

    norm_type = hp.Choice("%s_normalization"%prefix, ["none", "batch_norm", "layer_norm"], default="none")
    norm_before_after = hp.Choice("%s_norm_before_after"%prefix, ["before", "after"],
                    parent_name="%s_normalization"%prefix, parent_values=["batch_norm", "layer_norm"])

    return hidden_layers, activation, reg_type, l2, dropout, norm_type, norm_before_after



class DeepKoopman(tf.keras.Model):

    def __init__(self, hp, input_dim, total_steps, steps_for_loss, delta_t):
        """Create a Koopman network that encodes, advances in time, and decodes.

        Arguments:
            hp -- Kerastuner HyperParameters object
            input_dim -- number of variables in input data
            total_steps -- the total number of time steps in each example
            steps_for_loss -- number of time steps to optimize (max = total_steps - 1)
            delta_t -- amount of simulation time between each step (units are irrelevant)

        Returns:

        Side effects:
            None
        """
        super(DeepKoopman, self).__init__()

        self.delta_t = tf.constant(delta_t, name="delta_t", dtype=tf.float64)

        self.total_steps = total_steps
        self.steps_for_loss = min(steps_for_loss, total_steps - 1)

        # Choose size of compressed manifold.
        self.num_complex_pairs = hp.Int("omega_num_complex_pairs", 0, 5, default=1)
        self.num_real = hp.Int("omega_num_real", 0, 5, default=0)
        # 2 coordinates for each complex value and one coordinate for each real value.
        compressed_dim = self.num_complex_pairs * 2 + self.num_real

        # Choose loss weights
        self.L_inf_loss_lambda = 10.0 ** hp.Int("L_inf_loss_lambda_exponent", -10, -6)
        self.linearity_loss_lambda = 1.0
        self.reconstruction_loss_lambda = 10.0 ** hp.Int("reconstruction_loss_lambda_exponent", -4, 0)

        # Choose whether to start by only optimizing for reconstruction.
        self.times_total_loss_called = 0 # The number of times self.total_loss() has been called
        self.times_to_return_only_recon_loss = hp.Int("times_recon_loss_only", 1, 10000, sampling="log", default=100)


        # Choose structure of encoder and decoder hidden layers.
        hidden_layers, activation, reg_type, l2, dropout, norm_type, norm_before_after = mlp_hyperpams(hp, "enc_dec")
        encoder_widths = [input_dim] + hidden_layers + [compressed_dim]
        decoder_widths = [compressed_dim] + list(reversed(hidden_layers)) + [input_dim]

        self.encoder = mlp(encoder_widths, act_type=activation, reg_type=reg_type,
            l2=l2, dropout=dropout, norm_type=norm_type, norm_before_after=norm_before_after,
            name="encoder")

        self.decoder = mlp(decoder_widths, act_type=activation, reg_type=reg_type,
            l2=l2, dropout=dropout, norm_type=norm_type, norm_before_after=norm_before_after,
            name="decoder")


        self.omega_nets_complex, self.omega_nets_real = self.create_omega_nets(hp)

    def create_omega_nets(self, hp):
        """Create the auxiliary (omega) network(s), which have ycoords as input and output omegas (parameters for L).

        Arguments:
            hp -- Kerastuner hyperparameters object

        Returns:
            omega_nets_complex -- list,  complex conjugate pair omega (auxiliary) network(s)
            omega_nets_real -- list, real omega (auxiliary) network(s)

        Side effects:
            None
        """

        hidden_layers, activation, reg_type, l2, dropout, norm_type, norm_before_after = mlp_hyperpams(hp, "omega")
        omega_complex_widths = [1, ] + hidden_layers + [2, ]
        omega_real_widths = [1, ] + hidden_layers + [1, ]


        omega_nets_complex = []
        omega_nets_real = []

        for j in np.arange(self.num_complex_pairs):
            omega_nets_complex.append(
                mlp(omega_complex_widths, act_type=activation, reg_type=reg_type,
                    l2=l2, dropout=dropout, norm_type=norm_type, norm_before_after=norm_before_after,
                    name='OmegaComplex_%d' % (j + 1))
            )

        for j in np.arange(self.num_real):
            omega_nets_real.append(
                mlp(omega_real_widths, act_type=activation, reg_type=reg_type,
                    l2=l2, dropout=dropout, norm_type=norm_type, norm_before_after=norm_before_after,
                    name='OmegaReal_%d' % (j + 1))
            )

        return omega_nets_complex, omega_nets_real


    def call(self, inputs):
        """Call the model

        Arguments:
            inputs -- input tensor, shape = [batch_size, num_shifts, num_vars]

        Returns:
            y -- list, output of decoder applied to each shift: encoded_x[0], K*encoded_x[0], K^2*encoded_x[0], ..., length num_shifts + 1
                    shape = [batch_size, self.steps_for_loss+1, self.input_dim]

        Side effects:
            None

        """

        x_encoded = self.encoder_apply(inputs)
        encoded_layer = x_encoded[0,...]
        omegas = self.omega_net_apply(encoded_layer)

        y = []
        # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
        y.append(self.decoder(encoded_layer))

        # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
        advanced_layer = varying_multiply(encoded_layer, omegas, self.delta_t, self.num_real,
                                          self.num_complex_pairs)

        for j in range(self.steps_for_loss):
            # considering penalty on subset of yk+1, yk+2, yk+3, ...
            y.append(self.decoder(advanced_layer))

            omegas = self.omega_net_apply(advanced_layer)
            advanced_layer = varying_multiply(advanced_layer, omegas, self.delta_t, self.num_real,
                                              self.num_complex_pairs)

        if len(y) != self.steps_for_loss + 1:
            raise ValueError("messed up looping over shifts! %d != %d"%(len(y), self.steps_for_loss + 1))

        return tf.stack(y, axis=1, name="decoded_predictions")

    def encoder_apply(self, x):
        """Apply an encoder to data x.

        Arguments:
            x -- placeholder for input

        Returns:
            y -- output of encoder network applied to each time shift in input x,
                shape = [self.steps_for_loss+1, batch_size, self.input_dim]

        Side effects:
            None
        """
        y = [self.encoder(x[:, j, :]) for j in range(self.steps_for_loss + 1)]
        return tf.stack(y, axis=0, name="encoded_steps")

    def omega_net_apply(self, ycoords):
        """Apply the omega (auxiliary) network(s) to the y-coordinates.

        Arguments:
            ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k

        Returns:
            omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords

        Side effects:
            None
        """
        omegas = []
        for j in np.arange(self.num_complex_pairs):
            ind = 2 * j
            pair_of_columns = ycoords[:, ind:ind + 2]
            radius_of_pair = tf.reduce_sum(tf.square(pair_of_columns), axis=1, keepdims=True)
            omegas.append(self.omega_nets_complex[j](radius_of_pair))
        for j in np.arange(self.num_real):
            ind = 2 * self.num_complex_pairs + j
            one_column = ycoords[:, ind]
            omegas.append(self.omega_nets_real[j](one_column[:, np.newaxis]))

        return omegas

    def recon_loss(self, x, y):
        """Computes autoencoder reconstruction loss mean squared error.
        """
        return tf.keras.losses.mean_squared_error(x[:, 0, :], y[:, 0, :])

    def prediction_loss(self, x, y):
        """ Computes dynamics/prediction loss mean squared error.
        """
        for j in range(self.steps_for_loss):
            # xk+1, xk+2, xk+3
            if j==0: loss = tf.keras.losses.mean_squared_error(x[:, j, :], y[:, j + 1, :])
            else:
                loss = loss + tf.keras.losses.mean_squared_error(x[:, j, :], y[:, j + 1, :])
        return loss / self.steps_for_loss

    def linearity_loss(self, x, y):
        """ Computes K linearity loss mean squared error.
        """
        count_shifts_middle = 0
        g_list = self.encoder_apply(x)
        # generalization of: next_step = tf.matmul(g_list[0], L_pow)
        omegas = self.omega_net_apply(g_list[0, ...])
        next_step = varying_multiply(g_list[0, ...], omegas, self.delta_t, self.num_real,
                                         self.num_complex_pairs)
        loss = None
        # multiply g_list[0] by L (j+1) times
        for j in range(self.steps_for_loss):
            if loss==None: loss = tf.keras.losses.mean_squared_error(g_list[count_shifts_middle + 1, ...], next_step)
            else: loss = loss + tf.keras.losses.mean_squared_error(g_list[count_shifts_middle + 1, ...], next_step)
            omegas = self.omega_net_apply(next_step)
            next_step = varying_multiply(next_step, omegas, self.delta_t, self.num_real,
                                             self.num_complex_pairs)

        return loss / self.steps_for_loss

    def linf_loss(self, x, y):
        """ Computes inf norm on autoencoder loss and one-step prediction loss.
        """
        Linf1_penalty = tf.norm(tf.norm(y[:, 0, :] - tf.squeeze(x[:, 0, :]), axis=1, ord=np.inf), ord=np.inf)
        Linf2_penalty = tf.norm(tf.norm(y[:, 1, :] - tf.squeeze(x[:, 1, :]), axis=1, ord=np.inf), ord=np.inf)
        return Linf1_penalty + Linf2_penalty

    def total_loss(self, x, y):
        """ Computes total loss.
            Total loss will either be reconstruction loss by itself, or all losses combined.
        """
        self.times_total_loss_called += 1
        if self.times_total_loss_called < self.times_to_return_only_recon_loss-1:
            return self.recon_loss(x, y)
        return self.prediction_loss(x, y) + \
            self.reconstruction_loss_lambda * self.recon_loss(x, y) + \
            self.linearity_loss_lambda * self.linearity_loss(x, y) + \
            self.L_inf_loss_lambda * self.linf_loss(x, y)


def form_complex_conjugate_block(omegas, delta_t):
    """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension = [None, 2, 2]

    2x2 Block is
    exp(mu * delta_t) * [cos(omega * delta_t), -sin(omega * delta_t)
                         sin(omega * delta_t), cos(omega * delta_t)]

    Arguments:
        omegas -- array of parameters for blocks. first column is freq. (omega) and 2nd is scaling (mu), size [None, 2]
        delta_t -- time step in trajectories from input data

    Returns:
        stack of 2x2 blocks, size [None, 2, 2], where first dimension matches first dimension of omegas

    Side effects:
        None
    """
    scale = tf.exp(omegas[:, 1] * delta_t)
    entry11 = tf.multiply(scale, tf.cos(omegas[:, 0] * delta_t))
    entry12 = tf.multiply(scale, tf.sin(omegas[:, 0] * delta_t))
    row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
    row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
    return tf.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    """Multiply y-coordinates on the left by matrix L, but let matrix vary.

    Arguments:
        y -- array of shape [None, k] of y-coordinates, where L will be k x k
        omegas -- list of arrays of parameters for the L matrices
        delta_t -- time step in trajectories from input data
        num_real -- number of real eigenvalues
        num_complex_pairs -- number of pairs of complex conjugate eigenvalues

    Returns:
        array same size as input y, but advanced to next time step

    Side effects:
        None
    """
    complex_list = []

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = tf.stack([y[:, ind:ind + 2], y[:, ind:ind + 2]], axis=2)  # [None, 2, 2]
        L_stack = form_complex_conjugate_block(omegas[j], delta_t)
        elmtwise_prod = tf.multiply(ystack, L_stack)
        complex_list.append(tf.reduce_sum(elmtwise_prod, 1))

    if len(complex_list):
        # each element in list output_list is shape [None, 2]
        complex_part = tf.concat(complex_list, axis=1)

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    real_list = []
    for j in np.arange(num_real):
        ind = 2 * num_complex_pairs + j
        temp = y[:, ind]
        real_list.append(tf.multiply(temp[:, np.newaxis], tf.exp(omegas[num_complex_pairs + j] * delta_t)))

    if len(real_list):
        real_part = tf.concat(real_list, axis=1)
    if len(complex_list) and len(real_list):
        return tf.concat([complex_part, real_part], axis=1)
    elif len(complex_list):
        return complex_part
    else:
        return real_part




class DeepKoopmanHyperModel(kerastuner.HyperModel):

    def __init__(self, input_dim, total_steps, steps_for_loss, delta_t):
        """Create a hyperparameter tuner for a deep Koopman network

        Arguments:
            input_dim -- number of variables in input data
            total_steps -- the total number of time steps in each example
            steps_for_loss -- number of time steps to optimize (max = total_steps - 1)
            delta_t -- amount of simulation time between each step (units are irrelevant)):

        Arguments:
            None
        """
        self.input_dim = input_dim
        self.total_steps = total_steps
        self.steps_for_loss = steps_for_loss
        self.delta_t = delta_t


    def build(self, hp):
        deep_koop = DeepKoopman(hp, self.input_dim, self.total_steps, self.steps_for_loss, self.delta_t)

        optimizer_name = hp.Choice(
            'optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
        optimizer = tf.keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = hp.Choice(
            'learning_rate', [0.1, 0.01, 0.001, 0.0001], default=0.0001)

        deep_koop.compile(
            optimizer=optimizer,
            loss=deep_koop.total_loss,
            metrics=[deep_koop.recon_loss, deep_koop.prediction_loss, deep_koop.linearity_loss, deep_koop.linf_loss],
            # run_eagerly=True, # for debugging
        )
        return deep_koop
