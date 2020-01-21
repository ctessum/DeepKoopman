import numpy as np
import tensorflow as tf

import helperfns

def mlp(widths, act_type="relu", name="mlp", l2=None):
    """ Create a Keras multi-layer perceptron (MLP) model.

    Arguments:
        widths: A list of widths of each layer.
        act_type: Activation type
        name: Name of model

    Returns:
        a Keras model

    Side effects:
        None
    """
    model = tf.keras.Sequential(name=name)

    for i in np.arange(len(widths)):
        model.add(tf.keras.layers.Dense(
            widths[i],
            name="%s_%d"%(name,i),
            activation=None if i == len(widths)-1 else act_type, # Apply last layer without any nonlinearity
            dtype=tf.float64,
            kernel_regularizer=tf.keras.regularizers.l2(l2) if l2 else None,
        ))

    return model

class DeepKoopman(tf.keras.Model):

    def __init__(self, params):
        """Create a Koopman network that encodes, advances in time, and decodes.

        Arguments:
            params -- dictionary of parameters for experiment

        Returns:

        Side effects:
            None

        Raises ValueError if len(y) is not len(params['shifts']) + 1
        """
        super(DeepKoopman, self).__init__()

        depth = int((params['d'] - 4) / 2)

        max_shifts_to_stack = helperfns.num_shifts_in_stack(params)
        self.params = params

        encoder_widths = params['widths'][0:depth + 2]  # n ... k
        self.encoder = mlp(encoder_widths, act_type=params["act_type"], l2=params['L2_lam'], name="encoder")

        self.omega_nets_complex, self.omega_nets_real = create_omega_nets(params)

        num_widths = len(self.params['widths'])
        decoder_widths = self.params['widths'][depth + 2:num_widths]  # k ... n
        self.decoder = mlp(decoder_widths, act_type=self.params["act_type"], l2=params['L2_lam'], name="decoder")

    def call(self, inputs):
        """Call the model

        Arguments:
            inputs -- input tensor, shape = [num_shifts, batch_size, num_vars]

        Returns:
            y -- list, output of decoder applied to each shift: g_list[0], K*g_list[0], K^2*g_list[0], ..., length num_shifts + 1
            g_list -- list, output of encoder applied to each shift in input x, length num_shifts_middle + 1

        Side effects:
            Adds more entries to params dict: num_encoder_weights, num_omega_weights, num_decoder_weights

        Raises ValueError if len(y) is not len(params['shifts']) + 1
        """
        g_list = self.encoder_apply(inputs, shifts_middle=self.params['shifts_middle'])
        omegas = self.omega_net_apply(g_list[0])

        y = []
        # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
        encoded_layer = g_list[0]
        depth = int((self.params['d'] - 4) / 2)
        self.params['num_decoder_weights'] = depth + 1
        y.append(self.decoder(encoded_layer))

        # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
        advanced_layer = varying_multiply(encoded_layer, omegas, self.params['delta_t'], self.params['num_real'],
                                          self.params['num_complex_pairs'])

        for j in np.arange(max(self.params['shifts'])):
            # considering penalty on subset of yk+1, yk+2, yk+3, ...
            if (j + 1) in self.params['shifts']:
                y.append(self.decoder(advanced_layer))

            omegas = self.omega_net_apply(advanced_layer)
            advanced_layer = varying_multiply(advanced_layer, omegas, self.params['delta_t'], self.params['num_real'],
                                              self.params['num_complex_pairs'])

        if len(y) != (len(self.params['shifts']) + 1):
            print("messed up looping over shifts! %r" % self.params['shifts'])
            raise ValueError(
                'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

        return y

    def encoder_apply(self, x, shifts_middle):
        """Apply an encoder to data x.

        Arguments:
            x -- placeholder for input
            shifts_middle -- number of shifts (steps) in x to apply encoder to for linearity loss

        Returns:
            y -- list, output of encoder network applied to each time shift in input x

        Side effects:
            None
        """
        y = []
        num_shifts_middle = len(shifts_middle)
        for j in np.arange(num_shifts_middle + 1):
            if j == 0:
                shift = 0
            else:
                shift = shifts_middle[j - 1]
            if isinstance(x, (list,)):
                x_shift = x[shift]
            else:
                x_shift = x[shift, :, :]
            y.append(self.encoder(x_shift))
        return y

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
        for j in np.arange(self.params['num_complex_pairs']):
            ind = 2 * j
            pair_of_columns = ycoords[:, ind:ind + 2]
            radius_of_pair = tf.reduce_sum(tf.square(pair_of_columns), axis=1, keepdims=True)
            omegas.append(self.omega_nets_complex[j](radius_of_pair))
        for j in np.arange(self.params['num_real']):
            temp_name = 'OR%d_' % (j + 1)
            ind = 2 * self.params['num_complex_pairs'] + j
            one_column = ycoords[:, ind]
            omegas.append(self.omega_nets_real[j](one_column[:, np.newaxis]))

        return omegas

    def recon_loss(self, x, y):
        """Computes autoencoder reconstruction loss mean squared error.
        """
        return tf.keras.losses.mean_squared_error(x[0, ...], y[0])

    def prediction_loss(self, x, y):
        """ Computes dynamics/prediction loss mean squared error.
        """
        for j in np.arange(self.params['num_shifts']):
            shift = self.params['shifts'][j]
            # xk+1, xk+2, xk+3
            if j==0: loss = tf.keras.losses.mean_squared_error(x[shift, ...], y[j + 1])
            else:
                loss = loss + tf.keras.losses.mean_squared_error(x[shift, ...], y[j + 1])
        return loss / self.params['num_shifts']

    def linearity_loss(self, x, y):
        """ Computes K linearity loss mean squared error.
        """
        count_shifts_middle = 0
        g_list = self.encoder_apply(x, shifts_middle=self.params['shifts_middle'])
        # generalization of: next_step = tf.matmul(g_list[0], L_pow)
        omegas = self.omega_net_apply(g_list[0])
        next_step = varying_multiply(g_list[0], omegas, self.params['delta_t'], self.params['num_real'],
                                         self.params['num_complex_pairs'])
        loss = None
        # multiply g_list[0] by L (j+1) times
        for j in np.arange(max(self.params['shifts_middle'])):
            if (j + 1) in self.params['shifts_middle']:
                if loss==None: loss = tf.keras.losses.mean_squared_error(g_list[count_shifts_middle + 1], next_step)
                else: loss = loss + tf.keras.losses.mean_squared_error(g_list[count_shifts_middle + 1], next_step)
                count_shifts_middle += 1
            omegas = self.omega_net_apply(next_step)
            next_step = varying_multiply(next_step, omegas, self.params['delta_t'], self.params['num_real'],
                                             self.params['num_complex_pairs'])

        return loss / self.params['num_shifts_middle']

    def linf_loss(self, x, y):
        """ Computes inf norm on autoencoder loss and one-step prediction loss.
        """
        Linf1_penalty = tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf)
        Linf2_penalty = tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf)
        return Linf1_penalty + Linf2_penalty

    def total_loss(self, x, y):
        """ Computes total loss.
        """
        return self.params['recon_lam'] * self.recon_loss(x, y) + \
            self.params['recon_lam'] * self.prediction_loss(x, y) + \
            self.params['mid_shift_lam'] * self.linearity_loss(x, y) + \
            self.params['Linf_lam'] * self.linf_loss(x, y)


def form_complex_conjugate_block(omegas, delta_t):
    """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension [None, 2, 2]

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


def create_omega_nets(params):
    """Create the auxiliary (omega) network(s), which have ycoords as input and output omegas (parameters for L).

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        omega_nets_complex -- list,  complex conjugate pair omega (auxiliary) network(s)
        omega_nets_real -- list, real omega (auxiliary) network(s)

    Side effects:
        None
    """
    omega_nets_complex = []
    omega_nets_real = []

    for j in np.arange(params['num_complex_pairs']):
        omega_nets_complex.append(
            mlp(params['widths_omega_complex'], act_type=params["act_type"],
                    name='OmegaComplex_%d' % (j + 1), l2=params['L2_lam'])
        )

    for j in np.arange(params['num_real']):
        omega_nets_real.append(
            mlp(params['widths_omega_real'], act_type=params["act_type"],
                    name='OmegaReal_%d' % (j + 1), l2=params['L2_lam'])
        )

    return omega_nets_complex, omega_nets_real
