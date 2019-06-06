"""
Copyright (C) 2019 the LSNN team, TU Graz
"""

import numpy as np
from lsnn.toolbox.matplotlib_extension import strip_right_top_axis, raster_plot, hide_bottom_axis

def update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_data, batch=0, n_max_neuron_per_raster=300):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    fs = 12
    plt.rcParams.update({'font.size': fs})
    ylabel_x = -0.08
    ylabel_y = 0.5
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    ax_list[0].set_title("Target: " + str(plot_data['targets'][batch]))
    n_inhibitory_in_R = int((FLAGS.n_regular + FLAGS.n_adaptive) * (1 - FLAGS.proportion_excitatory))
    for k_data, data, d_name in zip(range(3),
                                    [plot_data['input_spikes'], plot_data['z_regular'],
                                     plot_data['z_adaptive']],
                                    ['Input', 'R', 'A']):

        ax = ax_list[k_data]
        if np.size(data) > 0:
            data = data[batch]
            if 'R' in d_name:
                # plot excitatory neurons
                exc_data = np.copy(data)
                exc_data[:,:n_inhibitory_in_R] = 0
                n_max = min(exc_data.shape[1], n_max_neuron_per_raster//2)
                cell_select = np.linspace(start=0, stop=exc_data.shape[1] - 1, num=n_max, dtype=int)
                exc_data = exc_data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
                raster_plot(ax, exc_data, colors='black')
                # plot inhibitory neurons
                inh_data = np.copy(data)
                inh_data[:,n_inhibitory_in_R:] = 0
                n_max = min(inh_data.shape[1], n_max_neuron_per_raster//2)
                cell_select = np.linspace(start=0, stop=inh_data.shape[1] - 1, num=n_max, dtype=int)
                inh_data = inh_data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
                raster_plot(ax, inh_data, colors='red')
                ax.set_yticklabels(['1', str(data.shape[-1])])
            else:
                n_max = min(data.shape[1], n_max_neuron_per_raster)
                cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
                data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
                if k_data == 0 and not FLAGS.crs_thr:
                    ax.imshow(data.T, aspect='auto', cmap='Greys')  # plot analog input differently
                    ax.set_yticklabels([])
                else:
                    raster_plot(ax, data)
                    ax.set_yticklabels(['1', str(data.shape[-1])])
            # y axis
            ax.set_ylabel(d_name, fontsize=fs)
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
            # bottom spine only needed for the bottom plot
            hide_bottom_axis(ax)


    try:
        # debug plot for psp-s or biases
        ax = ax_list[3]
        ax.set_ylabel('thresholds of A', fontsize=fs)
        threshold_data = plot_data['b_con'][batch]
        # subsample data to inlude only traces which achieve heigher threshold
        maxthr = np.amax(threshold_data, axis=0)
        mask = maxthr>np.mean(maxthr)*1.5
        ax.plot(threshold_data[:, mask], color='r', label='Output', alpha=0.5, linewidth=0.8)
        # set x axis limits
        ax.set_xlim([0, threshold_data.shape[0]])
        # bottom spine only needed for the bottom plot
        hide_bottom_axis(ax)
        # y axis
        ax.set_yticks([np.amin(threshold_data[:, mask]), np.amax(threshold_data[:, mask])])
        ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        # remove leading 0 and show only two decimal places for the threshold y ticks
        fig.canvas.draw()
        ylabs = [t.get_text()[1:4] for t in ax.get_yticklabels()]
        ax.set_yticklabels(ylabs)
    except Exception as e:
        print(e)

    # plot targets
    ax = ax_list[4]
    ax.set_yticks([0, 2, 4, 6, 8])
    classify_out = plot_data['out_plot'][batch]
    cax = ax.imshow(classify_out.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    cax.set_clim([0, 1])
    # inset legend
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbaxes = inset_axes(ax, width="3%", height="80%", loc=3)
    cbar = fig.colorbar(cax, cax=cbaxes, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '.5', '1'], fontsize=8)
    cbar.outline.set_edgecolor('white')
    cbaxes.tick_params(axis='both', colors='white')
    # y axis
    ax.set_ylabel('output Y', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    # x axis
    ax.set_xlabel('time in ms', fontsize=fs)
    plt.tight_layout()

    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.interactive_plot:
        plt.draw()
        plt.pause(1)