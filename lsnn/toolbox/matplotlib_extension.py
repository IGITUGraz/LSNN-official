"""
The Clear BSD License

Copyright (c) 2019 the LSNN team, institute for theoretical computer science, TU Graz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of LSNN nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = 'guillaume'

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib import cm
from collections import OrderedDict
from matplotlib.colors import LinearSegmentedColormap

def raster_plot(ax,spikes,linewidth=0.8,**kwargs):

    n_t,n_n = spikes.shape
    event_times,event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n,t in zip(event_ids,event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t])
    ax.set_yticks([0, n_n])

def strip_right_top_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def arrow_trajectory(ax,data,epsi=0,hdw=.03,lab='',fact=.8,color=(1.,1.,1.,1.),arrow_tick_steps=[],**kwargs):

    fc = tuple(np.clip(np.array(color) * fact,0,1.))

    ploted_lab = False

    X = data[:-1,:]
    dX = data[1:,:] - data[:-1,:]

    t0 = 0
    T = data.shape[0]-1

    if epsi > 0:
        while sum(dX[T-1]**2) / np.mean( np.sum(dX**2,axis=1)) < epsi: T = T-1
        while sum(dX[t0]**2) / np.mean(np.sum(dX**2,axis=1)) < epsi: t0 = t0+1

    ax.scatter(data[t0,0],data[t0,1],s=50,facecolor=fc,color=color,**kwargs)


    for t in np.arange(t0+1,T):
        x,y = X[t-1,:]
        dx,dy = dX[t-1,:]

        if t == T-1:
            headwidth = hdw
            head_length = hdw * 1.5
        elif t in arrow_tick_steps:
            headwidth = hdw
            head_length = hdw * 0.15
        else:
            headwidth = 0.
            head_length = 0.

        if dx != 0 or dy != 0:
            if ploted_lab:
                p = patches.FancyArrow(x, y, dx, dy,facecolor=color,edgecolor=fc,head_width=headwidth,head_length=head_length,**kwargs)
            else:
                ploted_lab = True
                p = patches.FancyArrow(x, y, dx, dy,facecolor=color,edgecolor=fc,head_width=headwidth,head_length=head_length,label=lab,**kwargs)
            ax.add_patch(p)


def hide_bottom_axis(ax):
    ax.spines['bottom'].set_visible(False)
    ax.set_xticklabels([])
    ax.get_xaxis().set_visible(False)
