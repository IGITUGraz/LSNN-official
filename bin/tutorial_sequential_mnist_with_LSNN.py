'''
The purpose of this script is to test the ALIF cell model and
provide a basic tutorial.

Out of the box you should get good performance (it uses a 20%
rewiring), please report any bug or unexpected performance.

One should get approximately:
- 40% accuracy in 100 iterations
- 60% in 200 iterations (about 30 minutes in our CPUs and faster with GPUs)
- you should eventually get above 90% with 20k ~ 30k iterations.
Successful runs should achieve up to 96% in 36k iterations (takes ~24h on GTX1080).

The Clear BSD License

Copyright (c) 2019 the LSNN team, institute for theoretical computer science, TU Graz
All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of LSNN nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lsnn.toolbox.file_saver_dumper_no_h5py import save_file, get_storage_path_reference
from tutorial_sequential_mnist_plot import update_mnist_plot

from lsnn.spiking_models import tf_cell_to_savable_dict, exp_convolve, ALIF, LIF
from lsnn.toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper
from lsnn.toolbox.tensorflow_utils import tf_downsample
import json
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
tf.app.flags.DEFINE_string('resume', '', 'path to the checkpoint of the form "results/script_name/2018_.../session"')
tf.app.flags.DEFINE_string('model', 'LSNN', 'spiking network model to use: LSNN or LIF')
tf.app.flags.DEFINE_bool('save_data', True, 'whether to save simulation data in result folder')
tf.app.flags.DEFINE_bool('downsampled', False, 'whether to use the smaller downsampled mnist dataset of not')
##
tf.app.flags.DEFINE_integer('n_batch_train', 256, 'size of the training minibatch')
tf.app.flags.DEFINE_integer('n_batch_validation', 256, 'size of the validation minibatch')
tf.app.flags.DEFINE_integer('n_in', 80, 'number of input units')
tf.app.flags.DEFINE_integer('n_regular', 120, 'number of regular spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('n_adaptive', 100, 'number of adaptive spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target firing rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 36000, 'number of training iterations')
tf.app.flags.DEFINE_integer('n_delay', 10, 'maximum synaptic delay')
tf.app.flags.DEFINE_integer('n_ref', 5, 'number of refractory steps')
tf.app.flags.DEFINE_integer('lr_decay_every', 2500, 'decay learning rate every lr_decay_every steps')
tf.app.flags.DEFINE_integer('print_every', 400, 'frequency of validation')
tf.app.flags.DEFINE_integer('ext_time', 1, 'repeat factor to extend time of mnist task')
##
tf.app.flags.DEFINE_float('beta', 1.8, 'Scaling constant of the adaptive threshold')
# to solve a task successfully we usually set tau_a to be close to the expected delay / memory length needed
tf.app.flags.DEFINE_float('tau_a', 700, 'Adaptation time constant')
tf.app.flags.DEFINE_float('tau_v', 20, 'Membrane time constant of output readouts')
tf.app.flags.DEFINE_float('thr', 0.01, 'Baseline threshold voltage')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Base learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1e-1, 'regularization coefficient to target a specific firing rate')
tf.app.flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
tf.app.flags.DEFINE_float('proportion_excitatory', 0.75, 'proportion of excitatory neurons')
##
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Perform plotting')
tf.app.flags.DEFINE_bool('verbose', True, 'Print many info during training')
tf.app.flags.DEFINE_bool('neuron_sign', True,
                         "If rewiring is active, this will fix the sign of neurons (Dale's law)")
tf.app.flags.DEFINE_bool('crs_thr', True, 'Encode pixels to spikes with threshold crossing method')
# With simple grid search we found that setting rewiring to 12% yields optimal results
tf.app.flags.DEFINE_float('rewiring_connectivity', 0.12, 'max connectivity limit in the network (-1 turns off DEEP R)')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization used in rewiring (irrelevant without rewiring)')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative')
# Analog values are fed to only single neuron
if not FLAGS.crs_thr:
    FLAGS.n_in = 1

assert FLAGS.model in ['LSNN', 'LIF'], "Model must be LSNN or LIF"
assert not (FLAGS.model == 'LIF' and FLAGS.n_adaptive > 0), "LIF network can not contain adaptive neurons!"

# Define the flag object as dictionnary for saving purposes
_, storage_path, flag_dict = get_storage_path_reference(__file__, FLAGS, './results/', flags=False,
                                                        comment=len(FLAGS.comment) > 0)
if FLAGS.save_data:
    os.makedirs(storage_path, exist_ok=True)
    save_file(flag_dict, storage_path, 'flag', 'json')
    print('saving data to: ' + storage_path)
print(json.dumps(flag_dict, indent=4))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Fix the random seed if given as an argument
dt = 1.  # Time step is by default 1 ms
n_output_symbols = 10

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(FLAGS.proportion_excitatory * FLAGS.n_in) + 1
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(FLAGS.proportion_excitatory * (FLAGS.n_regular + FLAGS.n_adaptive)) + 1
    n_inhibitory = FLAGS.n_regular + FLAGS.n_adaptive - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not (FLAGS.neuron_sign == False): print(
        'WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Define the network
if FLAGS.model == 'LSNN':
    # We set beta == 0 to some of the neurons. Those neurons then behave like LIF neurons (without adaptation).
    # And this is how we achieve a mixture of LIF and ALIF neurons in the LSNN model.
    beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
    cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=FLAGS.tau_v, n_delay=FLAGS.n_delay,
                n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=FLAGS.tau_a, beta=beta, thr=FLAGS.thr,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                dampening_factor=FLAGS.dampening_factor,
                )
elif FLAGS.model == 'LIF':
    cell = LIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=FLAGS.tau_v, n_delay=FLAGS.n_delay,
               n_refractory=FLAGS.n_ref, dt=dt, thr=FLAGS.thr,
               rewiring_connectivity=FLAGS.rewiring_connectivity,
               in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
               dampening_factor=FLAGS.dampening_factor
               )
else:
    raise NotImplementedError("Unknown model: " + FLAGS.model)

# Generate input
input_pixels = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder

targets = tf.placeholder(dtype=tf.int64, shape=(None,),
                         name='Targets')  # Lists of target characters of the recall task


def find_onset_offset(y, threshold):
    """
    Given the input signal `y` with samples,
    find the indices where `y` increases and descreases through the value `threshold`.
    Return stacked binary arrays of shape `y` indicating onset and offset threshold crossings.
    `y` must be 1-D numpy arrays.
    """
    if threshold == 1:
        equal = y == threshold
        transition_touch = np.where(equal)[0]
        touch_spikes = np.zeros_like(y)
        touch_spikes[transition_touch] = 1
        return np.expand_dims(touch_spikes, axis=0)
    else:
        # Find where y crosses the threshold (increasing).
        lower = y < threshold
        higher = y >= threshold
        transition_onset = np.where(lower[:-1] & higher[1:])[0]
        transition_offset = np.where(higher[:-1] & lower[1:])[0]
        onset_spikes = np.zeros_like(y)
        offset_spikes = np.zeros_like(y)
        onset_spikes[transition_onset] = 1
        offset_spikes[transition_offset] = 1

        return np.stack((onset_spikes, offset_spikes))


def get_data_dict(batch_size, type='train'):
    """
    Generate the dictionary to be fed when running a tensorflow op.
    """
    if type == 'test':
        input_px, target_oh = mnist.test.next_batch(batch_size, shuffle=False)
    elif type == 'validation':
        input_px, target_oh = mnist.validation.next_batch(batch_size)
    elif type == 'train':
        input_px, target_oh = mnist.train.next_batch(batch_size)
    else:
        raise ValueError("Wrong data group: " + str(type))

    target_num = np.argmax(target_oh, axis=1)

    if FLAGS.ext_time > 1:
        input_px = np.repeat(input_px, FLAGS.ext_time, axis=1)

    if FLAGS.crs_thr:
        # GENERATE THRESHOLD CROSSING SPIKES
        thrs = np.linspace(0, 1, FLAGS.n_in // 2)  # number of input neurons determins the resolution
        spike_stack = []
        for img in input_px:  # shape img = (784)
            Sspikes = None
            for thr in thrs:
                if Sspikes is not None:
                    Sspikes = np.concatenate((Sspikes, find_onset_offset(img, thr)))
                else:
                    Sspikes = find_onset_offset(img, thr)
            Sspikes = np.array(Sspikes)  # shape Sspikes = (31, 784)
            Sspikes = np.swapaxes(Sspikes, 0, 1)
            spike_stack.append(Sspikes)
        spike_stack = np.array(spike_stack)
        # add output cue neuron, and expand time for two image rows (2*28)
        out_cue_duration = 2 * 28 * FLAGS.ext_time
        spike_stack = np.lib.pad(spike_stack, ((0, 0), (0, out_cue_duration), (0, 1)), 'constant')
        # output cue neuron fires constantly for these additional recall steps
        spike_stack[:, -out_cue_duration:, -1] = 1
    else:
        spike_stack = input_px
        spike_stack = np.expand_dims(spike_stack, axis=2)
        # # match input dimensionality (add inactive output cue neuron)
        # spike_stack = np.lib.pad(spike_stack, ((0, 0), (0, 0), (0, 1)), 'constant')

    # transform target one hot from batch x classes to batch x time x classes
    data_dict = {input_pixels: spike_stack, targets: target_num}
    return data_dict, input_px


if not FLAGS.crs_thr and FLAGS.downsampled:
    inputs = tf.reshape(input_pixels,[-1,28,28,1])
    inputs = tf.layers.average_pooling2d(inputs,pool_size=2,strides=2,name='DownSampleWithPool',padding='same')
    inputs = tf.reshape(inputs,[-1,14 * 14,1])
else:
    inputs = input_pixels

outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
if FLAGS.model == 'LSNN':
    z, v, b = outputs
else:
    z = outputs
z_regular = z[:, :, :FLAGS.n_regular]
z_adaptive = z[:, :, FLAGS.n_regular:]

with tf.name_scope('ClassificationLoss'):
    psp_decay = np.exp(-dt / FLAGS.tau_v)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
    psp = exp_convolve(z, decay=psp_decay)
    n_neurons = z.get_shape()[2]

    # Define the readout weights
    if 0 < FLAGS.rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
                                                         FLAGS.rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])
    b_out = tf.get_variable(name='out_bias', shape=[n_output_symbols], initializer=tf.zeros_initializer())

    # Define the loss function
    out = einsum_bij_jk_to_bik(psp, w_out) + b_out

    if FLAGS.crs_thr:
        outt = tf_downsample(out, new_size=(28+2) * FLAGS.ext_time, axis=1)  # 32 x 30 x 10
        Y_predict = outt[:, -1, :]  # shape batch x classes == n_batch x 10
    else:
        Y_predict = out[:, -1, :]  # shape batch x classes == n_batch x 10

    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)

    # Define the accuracy
    Y_predict_num = tf.argmax(Y_predict, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    regularization_f0 = FLAGS.reg_rate / 1000
    loss_regularization = tf.reduce_sum(tf.square(av - regularization_f0)) * FLAGS.reg

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)  # Op to decay learning rate

    loss = loss_regularization + loss_recall

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:

        train_step = rewiring_optimizer_wrapper(optimizer, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                FLAGS.rewiring_connectivity,
                                                global_step=global_step,
                                                var_list=tf.trainable_variables())
    else:
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

# Real-time plotting
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if FLAGS.resume:
    saver.restore(sess, FLAGS.resume)
    print("Model restored.")

# Open an interactive matplotlib window to plot in real time
if FLAGS.interactive_plot:
    plt.ion()
    fig, ax_list = plt.subplots(5, figsize=(6, 7.5), gridspec_kw={'wspace':0, 'hspace':0.2})

# Store some results across iterations
test_loss_list = []
test_loss_with_reg_list = []
test_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []

# Dictionaries of tensorflow ops to be evaluated simultaneously by a session
results_tensors = {'loss': loss,
                   'loss_reg': loss_regularization,
                   'loss_recall': loss_recall,
                   'accuracy': accuracy,
                   'av': av,
                   'learning_rate': learning_rate,

                   'w_in_val': cell.w_in_val,
                   'w_rec_val': cell.w_rec_val,
                   'w_out': w_out,
                   }
if FLAGS.model == 'LSNN':
    results_tensors['b_out'] = b_out

plot_result_tensors = {'input_spikes': input_pixels,
                       'z': z,
                       'psp': psp,
                       'out_plot': out_plot,
                       'Y_predict': Y_predict,
                       'z_regular': z_regular,
                       'z_adaptive': z_adaptive,
                       'targets': targets}
if FLAGS.model == 'LSNN':
    plot_result_tensors['b_con'] = b

t_train = 0
for k_iter in range(FLAGS.n_iter):

    # Decaying learning rate
    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0 and mnist.train._epochs_completed > 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Print some values to monitor convergence
    if np.mod(k_iter, FLAGS.print_every) == 0:

        val_dict, input_img = get_data_dict(FLAGS.n_batch_validation, type='validation')
        results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)

        if FLAGS.save_data:
            save_file(results_values, storage_path, 'results_values', 'pickle')
            save_file(plot_results_values, storage_path, 'plot_results_values', 'pickle')

        # Storage of the results
        test_loss_with_reg_list.append(results_values['loss_reg'])
        test_loss_list.append(results_values['loss_recall'])
        test_error_list.append(results_values['accuracy'])
        training_time_list.append(t_train)

        print(
            '''Iteration {}, epoch {} validation accuracy {:.3g} '''
                .format(k_iter, mnist.train._epochs_completed, test_error_list[-1],))


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(results_values['av'] * 1000)

        # some connectivity statistics
        rewired_ref_list = ['w_in_val', 'w_rec_val', 'w_out']
        non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
        sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
        empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)
        empirical_connectivities = [nz / size for nz, size in zip(non_zeros, sizes)]

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f} (over neurons)
            connectivity (total {:.3g})\t W_in {:.3g} \t W_rec {:.2g} \t\t w_out {:.2g}
            number of non zero weights \t W_in {}/{} \t W_rec {}/{} \t w_out {}/{}

            classification loss {:.2g} \t regularization loss {:.2g}
            learning rate {:.2g} \t training op. time {:.2g}s
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                empirical_connectivity,
                empirical_connectivities[0], empirical_connectivities[1], empirical_connectivities[2],
                non_zeros[0], sizes[0],
                non_zeros[1], sizes[1],
                non_zeros[2], sizes[2],
                results_values['loss_recall'], results_values['loss_reg'],
                results_values['learning_rate'], t_train,
            ))

        # Save files result
        if FLAGS.save_data:
            results = {
                'error': test_error_list[-1],
                'loss': test_loss_list[-1],
                'loss_with_reg': test_loss_with_reg_list[-1],
                'loss_with_reg_list': test_loss_with_reg_list,
                'error_list': test_error_list,
                'loss_list': test_loss_list,
                'time_to_ref': time_to_ref_list,
                'training_time': training_time_list,
                'tau_delay_list': tau_delay_list,
                'flags': flag_dict,
            }
            save_file(results, storage_path, 'results', file_type='json')

        if FLAGS.interactive_plot:
            update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values)

    # train
    t0 = time()
    train_dict, input_img = get_data_dict(FLAGS.n_batch_train, type='train')
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    t_train = time() - t0

if FLAGS.interactive_plot:
    update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values)


if FLAGS.save_data:
    # Save the tensorflow graph
    saver.save(sess, os.path.join(storage_path, 'session'))
    saver.export_meta_graph(os.path.join(storage_path, 'graph.meta'))
    print("Network meta graph and session saved. Now testing...")

    # Testing
    test_errors = []
    n_test_batches = (mnist.test.num_examples//FLAGS.n_batch_validation) + 1
    for i in range(n_test_batches):  # cover the whole test set
        test_dict, input_img = get_data_dict(FLAGS.n_batch_validation, type='test')

        results_values, plot_results_values, in_spk, spk, targets_np = sess.run(
            [results_tensors, plot_result_tensors, input_pixels, z, targets],
            feed_dict=test_dict)
        test_errors.append(results_values['accuracy'])

    print('''Statistics on the test set: average accuracy {:.3g} +- {:.3g} (averaged over {} test batches of size {})'''
          .format(np.mean(test_errors), np.std(test_errors), n_test_batches, FLAGS.n_batch_validation))
    plot_results_values['test_imgs'] = np.array(input_img)
    save_file(plot_results_values, storage_path, 'plot_results_values', 'pickle')
    save_file(results_values, storage_path, 'results_values', 'pickle')

    # Save files result
    results = {
        'test_errors': test_errors,
        'test_errors_mean': np.mean(test_errors),
        'test_errors_std': np.std(test_errors),
        'error': test_error_list[-1] if test_error_list else None,
        'loss': test_loss_list[-1] if test_loss_list else None,
        'loss_with_reg': test_loss_with_reg_list[-1],
        'loss_with_reg_list': test_loss_with_reg_list,
        'error_list': test_error_list,
        'loss_list': test_loss_list,
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'flags': flag_dict,
    }

    save_file(results, storage_path, 'results', file_type='json')

    if FLAGS.interactive_plot:
        for i in range(min(8, FLAGS.n_batch_validation)):
            update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values, batch=i)
            fig.savefig(os.path.join(storage_path, 'figure_TEST_' + str(i) + '.pdf'), format='pdf')
            plt.show()
            plt.ioff()
del sess
