"""
Copyright (C) 2019 the LSNN team, TU Graz
"""

import tensorflow as tf
import numpy as np
import numpy.random as rd
import numpy.linalg as la
import matplotlib.pyplot as plt


def balance_matrix_per_neuron(M):
    M = M.copy()
    n_in, n_out = M.shape

    for k in range(n_out):
        # Change only non zero synapses to keep as much zeros as possible
        e_act = M[:, k] > 0
        i_act = M[:, k] < 0
        if np.sum(i_act) == 0:
            M[:, k] = 0
            print(
                'Warning: Neuron {} has not incoming synpases from inhibitory neurons. Setting all incoming weights to 0 to avoid un-balanced behaviour.'.format(
                    k))
        if np.sum(e_act) == 0:
            M[:, k] = 0
            print(
                'Warning: Neuron {} has not incoming synpases from excitatory neurons. Setting all incoming weights to 0 to avoid un-balanced behaviour.'.format(
                    k))

        s_e = M[e_act, k].sum()
        s_i = M[i_act, k].sum()

        # Add a small portion to compensate if the mean is not balanced
        if s_e + s_i < 0:
            M[e_act, k] += np.abs(s_e + s_i) / np.sum(e_act)
        else:
            M[i_act, k] -= np.abs(s_e + s_i) / np.sum(i_act)

        sum_check = M[:, k].sum()
        assert sum_check ** 2 < 1e-5, 'Mismatch of row balancing for neuron {}, sum is {} with on exci {} and inhib {}'.format(
            k, sum_check, s_e, s_i)

    return M


def max_eigen_value_on_unit_circle(w):
    vals = np.abs(la.eig(w)[0])
    factor = 1. / np.max(vals)
    return w * factor, factor


def random_sparse_signed_matrix(neuron_sign, p=1., balance_zero_mean_per_neuron=True, n_out=None):
    '''
    Provide a good initialization for a matrix with restricted sign.
    This is a personal recipe.

    :param neuron_sign:
    :param p:
    :param balance_zero_mean_per_neuron:
    :param n_out:
    :return:
    '''
    E = neuron_sign > 0
    I = neuron_sign < 0
    n = neuron_sign.__len__()

    if n_out is None:
        n_out = n

    # Random numbers
    is_con = rd.rand(n, n) < p

    theta = np.abs(rd.randn(n, n))
    theta = (2 * is_con - 1) * theta

    sign = np.tile(np.expand_dims(neuron_sign, 1), (1, n))
    w = lambda theta, sign: (theta) * (theta > 0) * sign
    _w = w(theta, sign)

    if (np.sum(I) > 0):
        # Normalize a first time, but this is obsolete if the stabilization happens also on a single neuron basis
        val_E = np.sum(_w[E, :])
        val_I = - np.sum(_w[I, :])
        assert val_I > 0 and val_E > 0, 'Sign error'
        theta[I, :] *= val_E / val_I
        _w = w(theta, sign)

        if balance_zero_mean_per_neuron:
            w_balanced = balance_matrix_per_neuron(_w)
            theta[theta > 0] = np.abs(w_balanced[theta > 0])
            _w = w(theta, sign)
            assert (_w[np.logical_not(is_con)] == 0).all(), 'Balancing the neurons procuded a sign error'

    else:
        print("Warning: no inhibitory neurons detected, no balancing is performed")

    # Normalize to scale the eigenvalues
    _, factor = max_eigen_value_on_unit_circle(_w)
    theta *= factor
    _w = w(theta, sign)

    assert (_w[E] >= 0).all(), 'Found negative excitatory weights'
    assert (_w[I] <= 0).all(), 'Found negative excitatory weights'

    if n_out is None:
        return w, sign, theta, is_con

    elif n < n_out:
        sel = np.random.choice(n, size=n_out)
    else:
        sel = np.arange(n_out)

    theta = theta[:, sel]
    sign = sign[:, sel]
    is_con = is_con[:, sel]

    return w(theta, sign), sign, theta, is_con


def test_random_sparse_signed_matrix():
    # Define parameter
    p = .33
    p_e = .75
    mean_E = .4
    std_E = 0
    n_in = 400
    neuron_sign = rd.choice([1, -1], n_in, p=[p_e, 1 - p_e])

    M1, M1_sign, M1_theta, M1_is_con = random_sparse_signed_matrix(neuron_sign=neuron_sign, p=p,
                                                                   balance_zero_mean_per_neuron=True)
    s1, _ = la.eig(M1)

    assert np.all(np.abs(M1[M1_is_con]) == M1_theta[M1_is_con])
    assert np.all(np.sign(M1) == M1_sign * M1_is_con)
    assert np.all(M1_is_con == (M1_theta > 0))

    M2, _, _, _ = random_sparse_signed_matrix(neuron_sign=neuron_sign, p=1., balance_zero_mean_per_neuron=True)
    M2 = M2 * (rd.rand(n_in, n_in) < p)
    s2, _ = la.eig(M2)

    fig, ax_list = plt.subplots(2)

    ax_list[0].set_title('Random sign constrained without neuron specific balance (p={:.3g})'.format(p))
    ax_list[1].set_title('Random sign constrained, probability mask taken after scaling')
    ax_list[0].scatter(s1.real, s1.imag)
    ax_list[1].scatter(s2.real, s2.imag)
    c = plt.Circle(xy=(0, 0), radius=1, edgecolor='r', alpha=.5)
    ax_list[0].add_artist(c)
    c = plt.Circle(xy=(0, 0), radius=1, edgecolor='r', alpha=.5)
    ax_list[1].add_artist(c)

    for ax in ax_list:
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])

    plt.show()


def sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(theta_list, ps, upper_bound_check=False):
    with tf.name_scope('NBreconnectGenerator'):
        theta_vals = [theta.read_value() for theta in theta_list]

        # Compute size and probability of connections
        nb_possible_connections_list = [tf.cast(tf.size(th), dtype=tf.float32) * p for th, p in zip(theta_list, ps)]
        total_possible_connections = tf.reduce_sum(nb_possible_connections_list)
        max_total_connections = tf.cast(total_possible_connections, dtype=tf.int32)
        sampling_probs = [nb_possible_connections / total_possible_connections \
                          for nb_possible_connections in nb_possible_connections_list]

        def nb_connected(theta_val):
            is_con = tf.greater(theta_val, 0)
            n_connected = tf.reduce_sum(tf.cast(is_con, tf.int32))
            return n_connected

        total_connected = tf.reduce_sum([nb_connected(theta) for theta in theta_vals])

        if upper_bound_check:
            assert_upper_bound_check = tf.Assert(tf.less_equal(total_connected, max_total_connections),
                                                 data=[max_total_connections, total_connected],
                                                 name='RewiringUpperBoundCheck')
        else:
            assert_upper_bound_check = tf.Assert(True,
                                                 data=[max_total_connections, total_connected],
                                                 name='SkippedRewiringUpperBoundCheck')

        with tf.control_dependencies([assert_upper_bound_check]):
            nb_reconnect = tf.maximum(0, max_total_connections - total_connected)

            sample_split = tf.distributions.Categorical(probs=sampling_probs).sample(nb_reconnect)
            is_class_i_list = [tf.equal(sample_split, i) for i in range(len(theta_list))]
            counts = [tf.reduce_sum(tf.cast(is_class_i, dtype=tf.int32)) for is_class_i in is_class_i_list]
            return counts


def compute_gradients_with_rewiring_variables(opt, loss, var_list):
    rewiring_w_list = tf.get_collection('Rewiring/Weights')
    rewiring_sign_list = tf.get_collection('Rewiring/Signs')
    rewiring_var_list = tf.get_collection('Rewiring/Variables')

    # generate the two sets of variables
    grads_and_vars = opt.compute_gradients(loss, var_list=var_list)

    # compute the gradients of rewired variables (disconnected vars have non zero gradients to avoid irregularities for optimizers with momentum)
    rewiring_gradient_list = tf.gradients(loss, rewiring_w_list)

    rewiring_gradient_list = [g * s if g is not None else None for g, s in
                              zip(rewiring_gradient_list, rewiring_sign_list)]
    rewiring_gradient_dict = dict([(v, g) for g, v in zip(rewiring_gradient_list, rewiring_var_list)])

    # OP to apply all gradient descent updates
    gathered_grads_and_vars = []
    for (g, v) in grads_and_vars:
        if v not in rewiring_var_list:
            gathered_grads_and_vars.append((g, v))
        else:
            gathered_grads_and_vars.append((rewiring_gradient_dict[v], v))

    return gathered_grads_and_vars


def get_global_connectivity_bound_assertion(rewiring_var_list, rewiring_connectivities):
    if np.isscalar(rewiring_connectivities): rewiring_connectivities = [rewiring_connectivities for _ in
                                                                        range(len(rewiring_var_list))]

    is_positive_theta_list = [tf.greater(th.read_value(), 0) for th in rewiring_var_list]

    n_connected_list = [tf.reduce_sum(tf.cast(is_pos, dtype=tf.float32)) for is_pos in is_positive_theta_list]
    size_list = [tf.size(is_pos) for is_pos in is_positive_theta_list]
    init_n_connected_list = [tf.cast(size, dtype=tf.float32) * p for size, p in
                             zip(size_list, rewiring_connectivities)]

    total_connected = tf.reduce_sum(n_connected_list)
    limit_connected = tf.reduce_sum(init_n_connected_list)

    check_connectivity = tf.Assert(total_connected <= limit_connected, [total_connected, limit_connected],
                                   name='CheckRewiringConnectivityBound')
    return check_connectivity


def rewiring_optimizer_wrapper(opt, loss, learning_rate, l1s, temperatures,
                               rewiring_connectivities, global_step=None,
                               var_list=None,
                               grads_and_vars=None):
    if var_list is None:
        var_list = tf.trainable_variables()

    # Select the rewired variable in the given list of variable to train
    rewiring_var_list = []
    for v in tf.get_collection('Rewiring/Variables'):
        if v in var_list:
            rewiring_var_list.append(v)

    if grads_and_vars is None:
        grads_and_vars = compute_gradients_with_rewiring_variables(opt, loss, var_list)
    else:
        grads_and_vars = grads_and_vars

    assert len(var_list) == len(grads_and_vars), 'Found {} elements in var_list and {} in grads_and_vars'.format(len(var_list),len(grads_and_vars))
    for v, gv in zip(var_list, grads_and_vars):
        assert v == gv[1]

    if np.isscalar(l1s): l1s = [l1s for _ in range(len(rewiring_var_list))]
    if np.isscalar(temperatures): temperatures = [temperatures for _ in range(len(rewiring_var_list))]
    if np.isscalar(rewiring_connectivities): rewiring_connectivities = [rewiring_connectivities for _ in
                                                                        range(len(rewiring_var_list))]

    is_positive_theta_list = [tf.greater(th, 0) for th in rewiring_var_list]
    with tf.control_dependencies(is_positive_theta_list):
        check_connectivity = get_global_connectivity_bound_assertion(rewiring_var_list, rewiring_connectivities)
        with tf.control_dependencies([check_connectivity]):
            gradient_check_list = [
                tf.check_numerics(g, message='Found NaN or Inf in gradients with respect to the variable ' + v.name) for
                (g, v) in grads_and_vars]

            with tf.control_dependencies(gradient_check_list):
                apply_gradients = opt.apply_gradients(grads_and_vars, global_step=global_step)

                if len(rewiring_var_list) == 0:
                    print('Warning: No variable to rewire are found by the rewiring optimizer wrapper')
                    return apply_gradients

                with tf.control_dependencies([apply_gradients]):
                    # This is to make sure that the algorithms does not reconnect synapses by mistakes,
                    # This can happen with optimizers like Adam
                    disconnection_guards = [tf.assign(var, tf.where(is_pos, var, tf.zeros_like(var))) for var, is_pos in
                                            zip(rewiring_var_list, is_positive_theta_list)]

                    with tf.control_dependencies(disconnection_guards):
                        rewiring_var_value_list = [th.read_value() for th in rewiring_var_list]

                        mask_connected = lambda th: tf.cast(tf.greater(th, 0), tf.float32)
                        noise_update = lambda th: mask_connected(th) * tf.random_normal(shape=tf.shape(th))

                        apply_regularization = [tf.assign_add(th, - learning_rate * mask_connected(th_) * l1 \
                                                              + tf.sqrt(2 * learning_rate * temp) * noise_update(th_))
                                                for th, th_, l1, temp in
                                                zip(rewiring_var_list, rewiring_var_value_list, l1s, temperatures)]

                        with tf.control_dependencies(apply_regularization):
                            number_of_rewired_connections = sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(
                                rewiring_var_list, rewiring_connectivities)

                            apply_rewiring = [rewiring(th, nb_reconnect=nb) for th, nb in
                                              zip(rewiring_var_list, number_of_rewired_connections)]
                            with tf.control_dependencies(apply_rewiring):
                                train_step = tf.no_op('Train')

    return train_step


def weight_sampler(n_in, n_out, p, dtype=tf.float32, neuron_sign=None, w_scale=1., eager=False):
    '''
    Returns a weight matrix and its underlying, variables, and sign matrices needed for rewiring.
    :param n_in:
    :param n_out:
    :param p0:
    :param dtype:
    :return:
    '''
    if eager:
        Variable = tf.contrib.eager.Variable
    else:
        Variable = tf.Variable

    with tf.name_scope('SynapticSampler'):

        nb_non_zero = int(n_in * n_out * p)

        # Gererate the random mask
        is_con_0 = np.zeros((n_in, n_out), dtype=bool)
        ind_in = rd.choice(np.arange(n_in), size=nb_non_zero)
        ind_out = rd.choice(np.arange(n_out), size=nb_non_zero)
        is_con_0[ind_in, ind_out] = True

        # Generate random signs
        if neuron_sign is None:

            theta_0 = np.abs(rd.randn(n_in, n_out) / np.sqrt(n_in))  # initial weight values
            theta_0 = theta_0 * is_con_0
            sign_0 = np.sign(rd.randn(n_in, n_out))
        else:
            assert np.size(neuron_sign) == n_in, 'Size of neuron_sign vector {}, for n_in {} expected'.format(
                np.size(neuron_sign), n_in)

            _, sign_0, theta_0, _ = random_sparse_signed_matrix(neuron_sign, n_out=n_out)
            theta_0 *= is_con_0

            # _, sign_0, theta_0, is_con_0 = random_sparse_signed_matrix(neuron_sign, p=p,
            #                                                            balance_zero_mean_per_neuron=True, n_out=n_out)

        # Define the tensorflow matrices
        th = Variable(theta_0 * w_scale, dtype=dtype, name='theta')
        w_sign = Variable(sign_0, dtype=dtype, trainable=False, name='sign')
        is_connected = tf.greater(th, 0, name='mask')
        w = tf.where(condition=is_connected, x=w_sign * th, y=tf.zeros((n_in, n_out), dtype=dtype), name='weight')

        # Add to collections to by pass and fetch them in the rewiring wrapper function
        tf.add_to_collection('Rewiring/Variables', th)
        tf.add_to_collection('Rewiring/Signs', w_sign)
        tf.add_to_collection('Rewiring/Weights', w)

        return w, w_sign, th, is_connected


def assert_connection_number(theta, targeted_number):
    '''
    Function to check during the tensorflow simulation if the number of connection in well defined after each simulation.
    :param theta:
    :param targeted_number:
    :return:
    '''
    th = theta.read_value()
    is_con = tf.greater(th, 0)

    nb_is_con = tf.reduce_sum(tf.cast(is_con, tf.int32))
    assert_is_con = tf.Assert(tf.equal(nb_is_con, targeted_number), data=[nb_is_con, targeted_number],
                              name='NumberOfConnectionCheck')
    return assert_is_con


def rewiring(theta, target_nb_connection=None, nb_reconnect=None, epsilon=1e-12, check_zero_numbers=False):
    '''
    The rewiring operation to use after each iteration.
    :param theta:
    :param target_nb_connection:
    :return:
    '''

    with tf.name_scope('rewiring'):
        th = theta.read_value()
        is_con = tf.greater(th, 0)

        reconnect_candidate_coord = tf.where(tf.logical_not(is_con), name='CandidateCoord')
        n_candidates = tf.shape(reconnect_candidate_coord)[0]

        if nb_reconnect is None:
            n_connected = tf.reduce_sum(tf.cast(is_con, tf.int32))
            nb_reconnect = target_nb_connection - n_connected

        nb_reconnect = tf.clip_by_value(nb_reconnect, 0, n_candidates)
        reconnect_sample_id = tf.random_shuffle(tf.range(n_candidates))[:nb_reconnect]
        reconnect_sample_coord = tf.gather(reconnect_candidate_coord, reconnect_sample_id, name='SelectedCoord')

        # Apply the rewiring
        reconnect_vals = tf.fill(dims=[nb_reconnect], value=epsilon, name='InitValues')
        reconnect_op = tf.scatter_nd_update(theta, reconnect_sample_coord, reconnect_vals, name='Reconnect')

        with tf.control_dependencies([reconnect_op]):
            if check_zero_numbers and target_nb_connection is not None:
                connection_check = assert_connection_number(theta=theta, targeted_number=target_nb_connection)
                with tf.control_dependencies([connection_check]):
                    return tf.no_op('Rewiring')
            else:
                return tf.no_op('Rewiring')


if __name__ == '__main__':
    test_random_sparse_signed_matrix()
