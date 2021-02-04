#!/usr/bin/env python3.6
#
# run_golden_model.py
#
# Gianna Paulin <pauling@iis.ee.ethz.ch>
# Francesco Conti <f.conti@unibo.it>
# Renzo Andri <andrire@iis.ee.ethz.ch>
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

import numpy as np
import argparse

from math import pow
import math
import sys

from rbe import RBE
from helper_functions import compare_models_numpy

np.set_printoptions(threshold=np.inf)

# MAIN FUNCTION
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run RBE golden model")

    ###
    ### Network parameters
    ###
    parser.add_argument("--kin", dest="k_in", type=int,
                        default="32",
                        help="NN parameter - number of input channels")
    parser.add_argument("--kout", dest="k_out", type=int,
                        default="32",
                        help="NN parameter - number of output channels")

    parser.add_argument("--hout", dest="h_out", type=int,
                        default=3,
                        help="NN parameter - spatial height of output feature map")
    parser.add_argument("--wout", dest="w_out", type=int,
                        default=3,
                        help="NN parameter - spatial width of output feature map")

    # TODO - more filter sizes?
    parser.add_argument("--fs", dest="fs", type=int,
                        default=3, choices=(1, 3),
                        help="NN parameter - filter size (x and y direction))")

    ###
    ### Hardware parameters
    ###
    # currently only 32 possible
    parser.add_argument("--tp", dest="tp", type=int,
                        default=32,
                        help="HW parameter - throughput parameter for BinConv")

    # currently only 4 possible
    parser.add_argument("--bs", dest="block_size", type=int,
                        default=4,
                        help="HW parameter - number of BinConvs in a block")

    # currently only 9 possible
    parser.add_argument("--colsize", dest="col_size", type=int,
                        default=9,
                        help="HW parameter - number of blocks in a column")
    # currently only 9 possible
    parser.add_argument("--nrcol", dest="nr_col", type=int,
                        default=9,
                        help="HW parameter - number of columns")

    parser.add_argument("--b_accum_width", dest="binconv_accum_width",  type=int,
                        default=16,
                        help="HW parameter - internal accumulation register width")
    parser.add_argument("--c_accum_width", dest="combination_accum_width",  type=int,
                        default=16,
                        help="HW parameter - internal accumulation register width")

    # quantizations of 1bit up to 8-bits possible
    parser.add_argument("--qw", dest="qw", type=int,
                        default=4, choices=(2, 3, 4, 5, 6, 7, 8),
                        help="HW parameter - activation quantization level (quantizations of 1bit up to 8-bits possible)")
    # quantizations of 1bit up to 8-bits possible in power-of-two steps
    parser.add_argument("--qa", dest="qa", type=int,
                        default=4, choices=(2, 3, 4, 5, 6, 7,8),
                        help="HW parameter - activation quantization level (quantizations of 1bit up to 8-bits possible in power-of-two steps)")
    # quantizations of 1bit up to 8-bits possible in power-of-two steps
    parser.add_argument("--qao", dest="qao", type=int,
                        default=4, choices=(2, 3, 4, 5, 6, 7, 8),
                        help="HW parameter - activation quantization level (quantizations of 1bit up to 8-bits possible in power-of-two steps)")


    # verify the bittrue golden model witht he simple golden model
    parser.add_argument("--vverify", dest="verify", type=int,
                        default=1, choices=(0,1),
                        help="Verification - SW golden model is verified with simple golden model")

    # use output quantization, batchnorm & relu
    parser.add_argument("--bn", dest="output_quant", type=int,
                        default=1, choices=(0,1),
                        help="Batchnorm - use batchnorm, relu etc")
    parser.add_argument("--relu", dest="relu", type=int,
                        default=1, choices=(0,1),
                        help="Use ReLU activation function.")

    # get arguments
    args = parser.parse_args()

    # create golden model with all HW and NN parameters
    rbe = RBE(nr_col=args.nr_col,
              col_size=args.col_size,
              block_size=args.block_size,
              tp=args.tp,
              qa=args.qa,
              qa_out=args.qao,
              qw=args.qw,
              fs=args.fs,
              h_out=args.h_out,
              w_out=args.w_out,
              k_out=args.k_out,
              k_in=args.k_in,
              binconv_accum_width=args.binconv_accum_width,
              combination_accum_width=args.combination_accum_width,
              output_quant=args.output_quant,
              relu=args.relu)

    # generate input stimuli
    rbe.generate_input_stimuli(shift_to_positive=True, signed_weights=True, signed_activations=False)

    # print the internal param of the gm
    rbe.print_overview()
    rbe.run_bittrue_shifted_network(verify_model=args.verify)

    if args.verify:
        y_comp = compare_models_numpy(rbe.y_golden, np.transpose(rbe.y_bittrue, (2,0,1)), True)
