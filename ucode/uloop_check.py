#!/usr/bin/env python3.6
#
# uloop_check.py
#
# Gianna Paulin <pauling@iis.ee.ethz.ch>
# Francesco Conti <f.conti@unibo.it>
#
# Copyright (C) 2019-2021 ETH Zurich, University of Bologna
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.sw.txt for details.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
from uloop_common import *
import math

# high-level loop
def iterate_hl_loop(TP, oh, ow, nof, nif, fs, qa, qw):



    ih = (oh - 1) + fs
    iw = (ow - 1) + fs

    FB = 5 # filter buffer size (FB*FB)
    BS = 4 # block size
    output_buffer_size = 3


    n_tiles_K_in = int(math.ceil(nif/TP))
    n_tiles_K_out = int(math.ceil(nof/TP))
    n_tiles_Hout = int(math.ceil(ih/FB))
    n_tiles_Wout = int(math.ceil(iw/FB))
    n_tiles_qa   = int(math.ceil(qa/BS))
    n_xpatches = n_tiles_Hout * n_tiles_Wout #* n_tiles_qa

    xpatch_size = FB*FB*BS*TP
    ypatch_size = output_buffer_size*output_buffer_size*TP

    # reset idices
    y_idx = 0
    W_idx = 0
    NQ_idx = 0
    x_idx = 0

    curr_idx = (0, 0, 0, 0, 0)

    if fs==3:
        n_w_stream_per_x_patch = qw
        w_stream_size = fs*fs*TP
        total_w_stream = n_w_stream_per_x_patch*w_stream_size
    else:
        n_w_stream_per_x_patch = fs
        w_stream_size = qw*TP
        total_w_stream = n_w_stream_per_x_patch*w_stream_size

    for i_major in range(n_tiles_Hout):
        for j_major in range(n_tiles_Wout):
            for k_out_major in range(n_tiles_K_out):
                for k_in_major in range(n_tiles_K_in):
                    for qa_major in range(n_tiles_qa):

                        # print(i_major, j_major, k_out_major, k_in_major, qa_major)
                        # print(n_tiles_Hout, n_tiles_Wout, n_tiles_K_out, n_tiles_K_in, n_tiles_qa)

                        W_idx = k_out_major*(n_tiles_K_in*(2+total_w_stream))-k_out_major*2 + k_in_major*(total_w_stream+2)
                        y_idx = i_major*n_tiles_Wout*n_tiles_K_out*ypatch_size + j_major*n_tiles_K_out*ypatch_size + k_out_major*ypatch_size
                        x_idx = i_major*n_tiles_Wout*(n_tiles_K_in*n_tiles_qa*xpatch_size) + j_major*(n_tiles_K_in*n_tiles_qa*xpatch_size) + k_in_major*(n_tiles_qa*xpatch_size) + qa_major*xpatch_size

                        NQ_idx = k_out_major * 32 *(32+16)

                        curr_idx = i_major, j_major, k_out_major, k_in_major, qa_major
                        yield W_idx, x_idx, NQ_idx, y_idx, curr_idx


VERBOSE = True


def uloop_check(TP, oh, ow, nof, nif, fs, qa, qw, verbose=VERBOSE):

    print("> Config TP=%d, oh=%d, ow=%d, nof=%d, nif=%d, fs=%d, qa=%d, qw=%d" % (TP, oh, ow, nof, nif, fs, qa, qw))

    ih = (oh - 1) + fs
    iw = (ow - 1) + fs


    FB = 5 # filter buffer size (FB*FB)
    BS = 4 # block size

    n_tiles_K_in = int(math.ceil(nif/TP))
    n_tiles_K_out = int(math.ceil(nof/TP))
    n_tiles_Hout = int(math.ceil(ih/FB))
    n_tiles_Wout = int(math.ceil(iw/FB))
    n_tiles_qa   = int(math.ceil(qa/BS))
    n_xpatches = n_tiles_Hout * n_tiles_Wout

    print("n_xpatches: ", n_xpatches)

    loops_range = [
        n_tiles_qa,
        n_tiles_K_in,
        n_tiles_K_out,
        n_xpatches
    ]

    if fs==3:
        stream_size_fs = TP*fs*qw

    else: # fs==1:
        stream_size_fs = TP*fs*fs*qw

    registers = [
        0,
        0,
        0,
        0,
        0,
        0,
        nif,
        nof,
        TP*FB*FB*4,
        TP*9,
        stream_size_fs, #TP*fs*qw, # or TP*fs*fs*qw
        TP*fs*fs*qw+2,
        32*(32+16),
        0
    ]

    loops_ops,code,mnem = uloop_load("code.yml")
    loops = uloop_get_loops(loops_ops, loops_range)

    err = 0
    idx  = []
    for j in range(NB_LOOPS):
        idx.append(0)
    state = (0,0,0,idx)
    busy = False
    execute = True
    # uloop_print_idx(state, registers)
    hidx = 0, 0, 0, 0, 0
    hl_loop = iterate_hl_loop(TP, oh, ow, nof, nif, fs, qa, qw)
    hW, hx, hNQ, hy, hidx = hl_loop.__next__()
    for i in range(0,1000000):
        new_registers = uloop_execute(state, code, registers)
        execute,end,busy,state = uloop_state_machine(loops, state, verbose=verbose)
        if execute:
            registers = new_registers
        if not busy:
            try:
                hW, hx, hNQ, hy, hidx = hW, hx, hNQ, hy, hidx = hl_loop.__next__()
            except StopIteration:
                pass
            if verbose:
                uloop_print_idx(state, registers)
            uW, ux, uNQ, uy = registers[0:4]
            if (hW != uW or hx != ux or hNQ != uNQ or hy != uy):
                if verbose:
                    print("  ERROR!!!")
                    print("  High-level: W=%d x=%d NQ=%d y=%d" % (hW, hx, hNQ, hy))
                    print("  uLoop:      W=%d x=%d NQ=%d y=%d" % (uW, ux, uNQ, uy))
                err += 1
        if end:
            break

    print(err, " errors", "!!!" if err > 0 else "")
    return err

for oh in range(3,12,3):
    for ow in range(3,12,3):
        for fs in (1,0):
            for nif in range(32, 64+32, 32):
                for qa in range(1,9):
                    for qw in range(1,9):
                        for nof in range(32, 64+32, 32):

                            err = uloop_check(
                                TP = 32,
                                fs = fs,
                                nof = nof,
                                nif = nif,
                                oh = oh,
                                ow = ow,
                                qa = qa,
                                qw = qw,
                                verbose = False
                            )
                            if err>0:
                                break
                        if err>0:
                            break
                    if err>0:
                        break
                if err>0:
                    break
            if err>0:
                break
        if err>0:
            break
    if err>0:
        break
if err>0:
    uloop_check(
        TP = 32,
        fs = fs,
        nof = nof,
        nif = nif,
        oh = oh,
        ow = ow,
        qa = qa,
        qw = qw,
        verbose = True
    )
