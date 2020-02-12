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

import tensorflow as tf

def einsum_bi_ijk_to_bjk(a,b):
    batch_size = tf.shape(a)[0]
    shp_a = a.get_shape()
    shp_b = b.get_shape()

    b_ = tf.reshape(b,(int(shp_b[0]), int(shp_b[1]) * int(shp_b[2])))
    ab_ = tf.matmul(a,b_)
    ab = tf.reshape(ab_,(batch_size,int(shp_b[1]),int(shp_b[2])))
    return ab

def einsum_bi_bij_to_bj(a,b):
    with tf.name_scope('Einsum-Bi-Bij-Bj'):
        a_ = tf.expand_dims(a,axis=1)
        a_b = tf.matmul(a_,b)
        ab = a_b[:,0,:]
    return ab

def einsum_bi_bijk_to_bjk(a,b):
    with tf.name_scope('Einsum-Bi-Bijk-Bjk'):
        a_ = a[:,:,None,None]
        a_b = a_ * b
        return tf.reduce_sum(a_b,axis=1)

def einsum_bij_jk_to_bik(a,b):
    try:
        n_b = int(a.get_shape()[0])
    except:
        n_b = tf.shape(a)[0]

    try:
        n_i = int(a.get_shape()[1])
    except:
        n_i = tf.shape(a)[1]

    try:
        n_j = int(a.get_shape()[2])
    except:
        n_j = tf.shape(a)[2]

    try:
        n_k = int(b.get_shape()[1])
    except:
        n_k = tf.shape(b)[1]

    a_ = tf.reshape(a,(n_b * n_i,n_j))
    a_b = tf.matmul(a_,b)
    ab = tf.reshape(a_b,(n_b,n_i,n_k))
    return ab


def einsum_bij_ki_to_bkj(a,b):

    # Write them as b k i j
    a_ = tf.expand_dims(a,axis=1)
    b_ = tf.expand_dims(b,axis=0)
    b_ = tf.expand_dims(b_,axis=3)

    ab = tf.reduce_sum(a_ * b_,axis=[2])
    return ab
