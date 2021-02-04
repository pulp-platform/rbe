#!/usr/bin/env python3.6
#
# rbe.py
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
from math import pow
import math
import sys

from helper_functions import *

np.set_printoptions(threshold=np.inf)

#deactivate bn for debugging
deactivate_bn = True

class RBE:
    """
    A class representing the golden model for the Reconfigurable Binary Engine


    Main Methods
    -------
    init_from_nemo(convLayer, batchnormLayer, ReLULayer, inputfm)
        Init RBE based on NEMO layers PACT_Conv2d, PACT_IntegerBatchNorm2d,
        PACT_IntegerAct and input FM.
    generate_input_stimuli(signed_weights=True, signed_activations=False,
     shift_to_positive=True, seed=0, distribution=None)
        Generates stimuli for python models and RTL model.
    import_stimuli_from_nemo(convLayer, batchnormLayer, ReLULayer, inputfm,
     signed_weights=True, signed_activations=False, shift_to_positive=True)
        Imports Stimuli based on NEMO layers PACT_Conv2d,
        PACT_IntegerBatchNorm2d, PACT_IntegerAct and input FM.
    import_stimuli(weights, kappa_bn, lambda_bn, shift_reqnt, act,
     signed_weights=True, signed_activations=False, shift_to_positive=True,
     isNemo=False)
        generates stimuli based on weights, and bn parameters kappa and lambda,
        and shift parameter.
    run_simple_golden_network(print_y=True)
        Runs golden model
    run_simple_shifted_network(print_y=True, verify_model=True)
        Runs a golden model with a seperate offset computation for supporting negative weights
    run_bittrue_shifted_network(verify_model=True)
        Runs bittrue model of the RBE
    """


    def __init__(self, nr_col=9, col_size=9, block_size=4, tp=32, qa=4, qw=4, qa_out=4, fs=3, h_out=3, w_out=3, k_out=32, k_in=32, binconv_accum_width=16, combination_accum_width=16, output_quant=True, relu=True):
        """
        Parameters
        ----------
        nr_col : int
            Number of columns (default is 9)
        col_size : int
            Number of blocks in a single column(default is 9)
        block_size : int
            number of BinConvs in a block (default is 4)
        tp : int
            The throughput paramter of the BinConvs (default is 32)
        qa : int
            The number of quantization bits of activations (default is 4)
        qw : int
            The number of quantization bits of weights (default is 4)
        qa_out : int
            The number of quantization bits of output activations (default is 4)
        fs : int
            The filter size of the network (default is 3)
        h_out : int
            The number of output pixel in y direction (default is 3)
            Non-multiple of 3 are filled with zeroes
        w_out : int
            The number of output pixel in x direction (default is 3)
            Non-multiple of 3 are filled with zeroes
        k_out : int
            The number of output channels (default is 32)
        k_in : int
            The number of input channels (default is 32)
        binconv_accum_width : int
            The bitwidth of the BinConv accumulator (default is 16)
        combination_accum_width : int
            The bitwidth of the column accumulator (default is 16)
        output_quant : bool
            If True, test output quantization
        relu : bool
            If True, ReLU is applied to the output (not supported by the HW)
        """

        # Hardware Parameters
        self.HW_BC_COLUMN_SIZE = col_size
        self.HW_NR_COLUMN      = nr_col
        self.HW_BC_BLOCK_SIZE  = block_size
        self.HW_TP             = tp

        self.F_BUFFER_SIZE = 5

        self.BINCONV_ACCUM_WIDTH     = binconv_accum_width
        self.COMBINATION_ACCUM_WIDTH = combination_accum_width

        # bounds for saturating adders
        self.BINCONV_ACCUM_UPPER_BOUND = pow(2, self.BINCONV_ACCUM_WIDTH - 1) - 1
        self.BINCONV_ACCUM_LOWER_BOUND = -pow(2, self.BINCONV_ACCUM_WIDTH - 1)

        self.COMBINATION_ACCUM_UPPER_BOUND = pow(2, self.COMBINATION_ACCUM_WIDTH - 1) - 1
        self.COMBINATION_ACCUM_LOWER_BOUND = -pow(2, self.COMBINATION_ACCUM_WIDTH - 1)

        # Network Parameters
        self.K_IN       = k_in
        self.K_OUT      = k_out
        self.K_OUT_int  = math.ceil(k_out/self.HW_TP)*self.HW_TP

        self.FS         = fs
        self.H_OUT      = h_out
        self.W_OUT      = w_out
        self.H_OUT_int  = math.ceil(h_out/3)*3 # round to the next multiple of 3
        self.W_OUT_int  = math.ceil(w_out/3)*3

        self.H_IN_int = (self.H_OUT_int - 1) + self.FS
        self.W_IN_int = (self.W_OUT_int - 1) + self.FS
        self.H_IN     = (self.H_OUT - 1) + self.FS
        self.W_IN     = (self.W_OUT - 1) + self.FS

        # Quantization configuration
        self.QA = qa
        self.QW = qw
        self.QA_OUT = qa_out
        self.RELU = relu
        self.isNEMO = False

        # Internal configuration values
        self.N_TILES_K_OUT = int(math.ceil(self.K_OUT_int / self.HW_TP))
        self.K_OUT_REST    = self.K_OUT % self.HW_TP

        self.N_TILES_K_IN = int(math.ceil(self.K_IN / self.HW_TP))
        self.K_IN_REST    = self.K_IN % self.HW_TP

        self.N_TILES_H_OUT = int(math.ceil(self.H_OUT_int/(self.HW_NR_COLUMN/3)))
        self.H_OUT_REST    = self.H_OUT_int % self.HW_NR_COLUMN

        self.N_TILES_W_OUT  = int(math.ceil(self.W_OUT_int/(self.HW_NR_COLUMN/3)))
        self.W_OUT_REST     = self.W_OUT_int % self.HW_TP

        self.N_TILES_H_IN = int(math.ceil(self.H_IN_int/self.F_BUFFER_SIZE))
        self.H_IN_int_REST    = self.H_IN_int % self.F_BUFFER_SIZE

        self.N_TILES_W_IN = int(math.ceil(self.W_IN_int/self.F_BUFFER_SIZE))
        self.W_IN_int_REST    = self.W_IN_int % self.F_BUFFER_SIZE

        self.N_TILES_QA    = int(math.ceil(self.QA/self.HW_BC_BLOCK_SIZE))
        self.QA_REST       = self.QA % self.HW_BC_BLOCK_SIZE

        self.N_TILES_QA_OUT = int(math.ceil(self.QA_OUT/self.HW_BC_BLOCK_SIZE))
        self.QA_OUT_REST    = self.QA_OUT % self.HW_BC_BLOCK_SIZE

        self.N_XPATCHES = self.N_TILES_H_OUT * self.N_TILES_W_OUT * self.N_TILES_QA * self.N_TILES_K_IN
        self.N_YPATCHES = self.N_TILES_H_OUT * self.N_TILES_W_OUT * self.N_TILES_QA_OUT * self.N_TILES_K_OUT

        # stimuli directories
        self.dir_golden_simple         = 'golden_simple/'
        self.dir_golden_simple_shifted = 'golden_simple_shifted/'
        self.dir_golden_bittrue        = 'golden_bittrue/'
        self.dir_compare               = 'golden_cmp/'

        self.output_quant = output_quant


    def init_from_nemo(convLayer, batchnormLayer, ReLULayer, inputfm):
        """
        Init RBE with parameters from NEMO.
        Usage: myRBE = RBE.init_from_nemo(convLayer, batchnormLayer, ReLULayer, inputfm)
        instead of myRBE = RBE(*args)

        Parameters
        ----------
        convLayer : PACT_Conv2d
            Quantized Convolution Layer
        batchnormLayer : PACT_QuantizedBatchNorm2d
            Quantized BatchNorm Layer
        ReLULayer : PACT_IntegerAct
            Quantized ReLU layer
        inputFM     : Tensor of size (1, K_IN, H_IN, W_IN)
            Input Activations/Feature Maps
        """
        # used hw configuration
        rbe_nr_col = 9
        rbe_col_size = 9
        rbe_block_size = 4
        rbe_tp = 32
        rbe_binconv_accum_width = 16
        rbe_combination_accum_width = 16
        output_quant = 1

        # assert initialized layers
        assert sum(convLayer.padding) == 0, "Zero Padding not supported"
        assert convLayer.kernel_size[0] == convLayer.kernel_size[1], "Non-symmetric kernels not supported"
        assert convLayer.deployment, "Not yet ready for deployment"

        # instantiate the rbe
        rbe = RBE(nr_col=rbe_nr_col,
          col_size=rbe_col_size,
          block_size=rbe_block_size,
          tp=rbe_tp,
          qa=convLayer.x_precision.get_bits(),
          qa_out=ReLULayer.precision.get_bits(),
          qw=convLayer.W_precision.get_bits()+1, # TODO acount for asymetry in training
          fs=convLayer.kernel_size[0],
          h_out=np.shape(inputfm)[2]-convLayer.kernel_size[0]+1,
          w_out=np.shape(inputfm)[3]-convLayer.kernel_size[0]+1,
          k_out=convLayer.out_channels,
          k_in=convLayer.in_channels,
          binconv_accum_width=rbe_binconv_accum_width,
          combination_accum_width=rbe_combination_accum_width,
          output_quant=output_quant)
        rbe.import_stimuli_from_nemo(convLayer, batchnormLayer, ReLULayer, inputfm)
        return rbe


    def print_overview(self):
        """Prints a list of all parameters """
        print("\n    #################################################")
        print("    ###       Reconfigurable Binary Engine        ###")
        print("    ###               Golden Model                ###")
        print("    #################################################")
        self.print_hw_parameters()
        self.print_nn_parameters()


    def print_hashline(self):
        """Prints a line of # """
        print("    #################################################")


    def print_hw_parameters(self):
        """Prints the HW parameters """
        print("""\n    # Hardware parameters
    HW_TP                   = {}\t\t
    HW_BC_BLOCK_SIZE        = {}\t\t
    HW_BC_COLUMN_SIZE       = {}\t\t
    HW_NR_COLUMN            = {}\t\t
    QA                      = {}\t\t
    QA_OUT                  = {}\t\t
    QW                      = {}\t\t
    BINCONV_ACCUM_WIDTH     = {}\t\t
    COMBINATION_ACCUM_WIDTH = {}\t\t
        """.format(self.HW_TP,
                   self.HW_BC_BLOCK_SIZE,
                   self.HW_BC_COLUMN_SIZE,
                   self.HW_NR_COLUMN,
                   self.QA,
                   self.QA_OUT,
                   self.QW,
                   self.BINCONV_ACCUM_WIDTH,
                   self.COMBINATION_ACCUM_WIDTH))


    def print_nn_parameters(self):
        """Prints the NN parameters """
        print("""    # Network parameters
    H_IN                    = {}\t\t
    W_IN                    = {}\t\t
    H_OUT                   = {}\t\t
    W_OUT                   = {}\t\t
    FS                      = {}\t\t
    K_IN                    = {}\t\t
    K_OUT                   = {}\t\t

    N_TILES_K_OUT           = {}\t\t
    N_TILES_K_IN            = {}\t\t
    N_TILES_H_OUT           = {}\t\t
    N_TILES_W_OUT           = {}\t\t
    N_TILES_H_IN            = {}\t\t
    N_TILES_W_IN            = {}\t\t
    N_TILES_QA              = {}\t\t
    N_TILES_QA_OUT          = {}\t\t
    N_XPATCHES              = {}\t\t
        """.format(self.H_IN_int,
                   self.W_IN_int,
                   self.H_OUT_int,
                   self.W_OUT_int,
                   self.FS,
                   self.K_IN,
                   self.K_OUT,
                   self.N_TILES_K_OUT,
                   self.N_TILES_K_IN,
                   self.N_TILES_H_OUT,
                   self.N_TILES_W_OUT,
                   self.N_TILES_H_IN,
                   self.N_TILES_W_IN,
                   self.N_TILES_QA,
                   self.N_TILES_QA_OUT,
                   self.N_XPATCHES))


    def generate_input_stimuli(self, signed_weights=True, signed_activations=False, shift_to_positive=True, seed=0, distribution=None):
        """
        Generates the stimuli for the network that should be running on the accelerator

        Parameters
        ----------
        signed_weights : bool
            True: the weights are generated from [-2**(QW-q)+1, 2**(QW-1)-1]
            False: the weights are generated from [0, 2**(QW)]
        signed_activations : bool
            True: the activations are generated from [-2**(QA-q)+1, 2**(QA-1)-1]
            False: the activations are generated from [0, 2**(QA)]
        shift_to_positive : bool
            True: if signed_weights=True => the signed weights are shifted to be
            positive, computed and the shift is reversed at the end of the computation
        seed : int
            give a seed for reproducibility (default=0: random seed is used)
        distribution : string
            if 'gaussian' (default), use normally distributed weights; else, uniformly distributed
        """

        # if seed is not set, randomly pick one
        if seed == 0: # Created random seed generation from system microtime
            import time
            mytime = time.time()
            mytime = round(100000*(mytime-round(mytime-0.5)))
            np.random.seed(mytime)
        else:
            np.random.seed(seed)

        # store stimuli configuration
        self.signed_weights    = signed_weights
        self.shift_to_positive = shift_to_positive

        # generate activations
        self.x = np.zeros(shape=(self.H_IN_int, self.W_IN_int, self.K_IN), dtype=int)

        if signed_activations:
            self.x[0:self.H_IN, 0:self.W_IN, :] = np.random.randint(-pow(2, self.QA - 1) + 1,
                                       pow(2, self.QA - 1),
                                       size=(self.H_IN, self.W_IN, self.K_IN), dtype=int)
        else:
            self.x[0:self.H_IN, 0:self.W_IN, :] = np.random.randint(0,
                                       pow(2, self.QA),
                                       size=(self.H_IN, self.W_IN, self.K_IN), dtype=int)
            # x = np.random.randint(-pow(2,QA-1)+1, 0, size=(H_IN, W_IN, K_IN)) # only neg numbers

        # generate weights
        if shift_to_positive:
            # qw_shifted = self.QW + 1
            self.Wmin = int(-pow(2, self.QW - 1) + 1)
            self.Wmax = int(pow(2, self.QW - 1) - 1)
        elif (signed_weights == True) and (shift_to_positive == False):
            self.Wmin = int(-pow(2, self.QW - 1) + 1)
            self.Wmax = int(pow(2, self.QW - 1) - 1)
        elif (signed_weights == False) and (shift_to_positive == False):
            self.Wmin = int(0)
            self.Wmax = int(pow(2, self.QW) - 1)
        # binary
        if self.QW == 1:
            self.Wmax = 1

        if distribution == 'gaussian':    # TODO with 2 bits a lot of 0s?
            mean = 0.5                    # compensate for np.floor bias
            std  = 0.27*np.abs(self.Wmax) # arbitrary, to have at least same tail values
            self.W = np.asarray(np.floor(np.random.normal(mean, std, size=(self.K_OUT, self.FS, self.FS, self.K_IN))).clip(self.Wmin, self.Wmax+1), dtype='int')
        else: # uniform
            self.W = np.random.randint(self.Wmin,
                                       self.Wmax+1,
                                       size=(self.K_OUT, self.FS, self.FS, self.K_IN))

        # generate abs and signs of weights
        self.Wabs = np.absolute(self.W)
        Wtmp = np.sign(self.W)
        Wtmp2 = np.where(Wtmp==0, 1, Wtmp)
        self.Wsigns = np.where(Wtmp2==-1, 0, Wtmp2)

        # generate shifted weights if necessary
        # generate abs and signs of shifted weights
        if shift_to_positive:
            self.Wshifted = np.add(self.W, -self.Wmin)
            self.Wsabs = np.absolute(self.Wshifted)
            Wstmp = np.sign(self.Wshifted)
            Wstmp2 = np.where(Wstmp==0, 1, Wstmp)
            self.Wssigns = np.where(Wstmp2==-1, 0, Wstmp2)

        # generate abs and signs of activations
        self.xabs = np.absolute(self.x)
        xtmp = np.sign(self.x)
        xtmp2 = np.where(xtmp==0, 1, xtmp)
        self.xsigns = np.where(xtmp2==-1, 0, xtmp2)

        # generate norm/requantization parameters
        self.kappa_bn    = np.random.randint(1, 2**16-1, size=(self.K_OUT,)) # absorbing also the mult_reqnt, real range INT-32
        self.lambda_bn   = np.random.randint(-2**31, 2**31, size=(self.K_OUT,)) # absorbing also the mult_reqnt, real range INT-32
        self.shift_reqnt = np.random.randint(0, 15, size=(1,))

        if(deactivate_bn):
            self.kappa_bn.fill(1)
            self.lambda_bn.fill(0)
            self.shift_reqnt.fill(0)

        self.write_file_normquant_param()


    def import_stimuli_from_nemo(self, convLayer, batchnormLayer, ReLULayer, inputfm, signed_weights=True, signed_activations=False, shift_to_positive=True):
        """
        Generates the stimuli for the network that should be running on the RBE
        accelerator based on NEMO layers.

        Parameters
        ----------
        convLayer : PACT_Conv2d
            Quantized Convolution Layer
        batchnormLayer : PACT_QuantizedBatchNorm2d
            Quantized BatchNorm Layer
        ReLULazer : PACT_IntegerAct
            Quantized ReLU layer
        inputFM     : Tensor of size (1, K_IN, H_IN, W_IN)
            Input Activations/Feature Maps
        signed_weights : bool
            True: the weights are in [-2**(QW-q)+1, 2**(QW-1)-1]
            False: the weights are in [0, 2**(QW)]
        signed_activations : bool
            True: the activations are in [-2**(QA-q)+1, 2**(QA-1)-1]
            False: the activations are in [0, 2**(QA)]
        shift_to_positive : bool
            True: if signed_weights=True => the signed weights are shifted to be
            positive, computed and the shift is reversed at the end of the computation
        """
        try:
            pact_integer_requantize
        except NameError:
            from nemo.quant.pact import pact_integer_requantize

        output = convLayer(inputfm)

        output_bn = batchnormLayer(output)
        lambda_bn = pact_integer_requantize(batchnormLayer.lamda, batchnormLayer.eps_lamda, batchnormLayer.eps_kappa*batchnormLayer.eps_in)
        kappa_bn = batchnormLayer.kappa

        output_bn_relu = ReLULayer(output_bn)
        eps_ratio_relu = (ReLULayer.D*ReLULayer.eps_in/ReLULayer.eps_out).round()
        shift_reqnt = np.int32(np.array([math.log2(ReLULayer.D)]))

        kappa_merged = kappa_bn*eps_ratio_relu
        lambda_merged = lambda_bn*eps_ratio_relu

        self.y_nemo = output_bn_relu[0] # batch=0 (no batching possible)
        self.import_stimuli(convLayer.weight, np.int64(kappa_merged.flatten()), np.int64(lambda_merged.flatten()), shift_reqnt, inputfm, signed_weights, signed_activations, shift_to_positive, True)


    def import_stimuli(self, weights, kappa_bn, lambda_bn, shift_reqnt, act, signed_weights=True, signed_activations=False, shift_to_positive=True, isNemo=False):
        """
        Generates the stimuli for the network that should be running on the RBE accelerator

        Parameters
        ----------
        weights : Tensor of size (K_OUT, K_IN, FS, FS)
        kappa_bn: Tensor of size (1, K_OUT, 1, 1)
            batch norm parameter kappa
        shift_reqnt : int
            Size of Shift before Requantization
        act     : Tensor of size (1, K_IN, H_IN, W_IN)
            Input Activations/Feature Maps
        signed_weights : bool
            True: the weights are in [-2**(QW-q)+1, 2**(QW-1)-1]
            False: the weights are in [0, 2**(QW)]
        signed_activations : bool
            True: the activations are in [-2**(QA-q)+1, 2**(QA-1)-1]
            False: the activations are in [0, 2**(QA)]
        shift_to_positive : bool
            True: if signed_weights=True => the signed weights are shifted to be
            positive, computed and the shift is reversed at the end of the computation
        """
        if not isNemo:
            del self.y_nemo

        self.isNEMO = isNemo

        # store stimuli configuration
        self.signed_weights    = signed_weights
        self.shift_to_positive = shift_to_positive

        # generate activations
        self.x = np.zeros(shape=(self.H_IN_int, self.W_IN_int, self.K_IN), dtype=int);
        self.x[0:self.H_IN, 0:self.W_IN, :] = np.transpose(act[0, :, :, :].detach(), (1,2,0))

        if not signed_activations:
            assert sum(sum(sum(self.x<0))) == 0, "Negative Values detected"

        # generate weights
        if shift_to_positive:
            self.Wmin = int(-pow(2, self.QW - 1))
            self.Wmax = int(pow(2, self.QW - 1) - 1)
        elif (signed_weights == True) and (shift_to_positive == False):
            self.Wmin = int(-pow(2, self.QW - 1))
            self.Wmax = int(pow(2, self.QW - 1) - 1)
        elif (signed_weights == False) and (shift_to_positive == False):
            self.Wmin = int(0)
            self.Wmax = int(pow(2, self.QW) - 1)

        assert sum(sum(sum(sum(weights < self.Wmin)))) == 0, "x<Wmin occurred"
        assert sum(sum(sum(sum(weights > self.Wmax)))) == 0, "x>Wmax occurred"

        assert np.shape(weights) == (self.K_OUT, self.K_IN, self.FS, self.FS), "weights have wrong dimension! ({},{},{},{}) expected.".format(self.K_OUT, self.K_IN, self.FS, self.FS)
        self.W = np.transpose(np.int64(weights.detach()), (0,2,3,1))

        # generate abs and signs of weights
        self.Wabs = np.absolute(self.W)
        Wtmp = np.sign(self.W)
        Wtmp2 = np.where(Wtmp==0, 1, Wtmp)
        self.Wsigns = np.where(Wtmp2==-1, 0, Wtmp2)

        # generate shifted weights if necessary
        # generate abs and signs of shifted weights
        if shift_to_positive:
            self.Wshifted = np.add(self.W, -self.Wmin)
            self.Wsabs = np.absolute(self.Wshifted)
            Wstmp = np.sign(self.Wshifted)
            Wstmp2 = np.where(Wstmp==0, 1, Wstmp)
            self.Wssigns = np.where(Wstmp2==-1, 0, Wstmp2)


        # generate abs and signs of activations
        self.xabs = np.absolute(self.x)
        xtmp = np.sign(self.x)
        xtmp2 = np.where(xtmp==0, 1, xtmp)
        self.xsigns = np.where(xtmp2==-1, 0, xtmp2)

        # generate norm/requantization parameters
        self.kappa_bn = kappa_bn
        self.lambda_bn   = lambda_bn
        self.shift_reqnt = shift_reqnt

        # write out normalization and quantization stimuli
        self.write_file_normquant_param()


    def write_file_normquant_param(self):
        """
            Writes the normalization and quantization parameter of the network into a
            stimuli file called`self.dir_golden_bittrue + 'stim_nq.h'
        """
        # Batch Normalization
        kappa_bn = self.kappa_bn    # mult
        lambda_bn = self.lambda_bn  # add

        nq_size = int(self.K_OUT * 48 / 8)

        # create and open file
        file = open(self.dir_golden_bittrue + 'stim_nq.h', "w+")

        file.write("#define STIM_NQ_SIZE 0x%02x \n" % nq_size)
        file.write("__attribute__((aligned(16))) volatile uint16_t stim_nq[STIM_NQ_SIZE] = {\n")

        # loop over output channels
        for k in range(self.K_OUT):

            file.write("0x%02x,\n" % kappa_bn[k])
            tmp = lambda_bn[k]
            tmp = tmp & 0x0000ffff
            file.write("0x%02x,\n" % tmp)

            tmp = lambda_bn[k]
            tmp = (tmp & 0xffff0000) >> 16
            file.write("0x%02x,\n" % tmp)

        file.write("}; \n")
        file.close()

        # RELU
        shift_reqnt = self.shift_reqnt[0]
        # print("shift_reqnt: ", shift_reqnt)

        # create and open file
        file = open(self.dir_golden_bittrue + 'stim_shift.h', "w+")

        file.write("#define SHFIT_REQNT 0x%02x \n" % shift_reqnt)
        file.close()


    def write_file_w_streams(self):
        """
            Writes out the created W_streams into the file `self.dir_golden_bittrue+'stim_W.h'`
        """
        W_streams = self.W_streams
        W_size  = int(self.N_TILES_K_OUT * self.N_TILES_K_IN * self.N_STREAMS_PER_XPATCH * self.HW_BC_COLUMN_SIZE * self.HW_TP/8)

        # open file
        file = open(self.dir_golden_bittrue+'stim_W.h',"w+")

        # write c header file related stuff
        file.write("#define STIM_W_SIZE 0x%02x \n" % W_size)
        file.write("__attribute__((aligned(16))) volatile uint32_t stim_W[STIM_W_SIZE] = {\n")

        if (self.FS ==3):
            for k_out_major in range(self.N_TILES_K_OUT):
                for k_in_major in range(self.N_TILES_K_IN):
                    for p in range(self.N_STREAMS_PER_XPATCH):
                        for c in range(self.HW_BC_COLUMN_SIZE):
                            bits = W_streams[k_out_major,k_in_major,p,c,0:32]
                            dec = bin2dec(bits)
                            file.write("0x%02x,\n" % dec)

        elif (self.FS == 1):
            for k_out_major in range(self.N_TILES_K_OUT):
                for k_in_major in range(self.N_TILES_K_IN):
                    for p in range(self.N_STREAMS_PER_XPATCH):
                        for c in range(self.QW):
                            bits = W_streams[k_out_major,k_in_major,p,c,0:32]
                            dec = bin2dec(bits)
                            file.write("0x%02x,\n" % dec)

        file.write("}; \n")
        file.close()


    def write_file_x_image(self):
        """
            Writes out the created x images into the file `self.dir_golden_bittrue+'stim_x.h'`
        """
        x_image = self.x_image
        x_size  = int(self.H_IN_int*self.W_IN_int*self.N_TILES_K_IN*self.N_TILES_QA*self.HW_BC_BLOCK_SIZE*self.HW_TP/8)

        file = open(self.dir_golden_bittrue+'stim_x.h',"w+")

        file.write("#define STIM_X_SIZE 0x%02x \n" % x_size)
        file.write("__attribute__((aligned(16))) volatile uint32_t stim_x[STIM_X_SIZE] = {\n")

        for i in range(self.H_IN_int):
            for k_in_major in range(self.N_TILES_K_IN):
                for qa_major in range(self.N_TILES_QA):
                    for b in range(self.HW_BC_BLOCK_SIZE):
                        for j in range(self.W_IN_int):
                            bits = x_image[i,j,k_in_major,qa_major,b,0:32]
                            dec = bin2dec(bits)
                            file.write("0x%02x,\n" % dec)

        file.write("}; \n")
        file.close()


    def write_file_y_image(self):
        """
            Writes out the created y images into the file `self.dir_golden_bittrue+'stim_y.h'`.
            Used as golden values.
            At the same time write out zeros into `self.dir_golden_bittrue+'y_dump.h'`. These values
            will be overwritten by the accelerator
        """
        y_image = self.y_image

        # number of bytes when aligning data with self.HW_BC_BLOCK_SIZE*self.HW_TP
        y_size  = int(self.H_OUT_int*self.W_OUT_int*self.N_TILES_K_OUT*self.N_TILES_QA_OUT*self.HW_BC_BLOCK_SIZE*self.HW_TP/8)

        file  = open(self.dir_golden_bittrue+'stim_y.h',"w+")
        filedh = open(self.dir_golden_bittrue+'y_dump.h',"w+")

        file.write("#define STIM_Y_SIZE 0x%02x \n" % y_size)
        file.write("__attribute__((aligned(16))) volatile uint32_t stim_y[STIM_Y_SIZE] = {\n")

        filedh.write("#define STIM_Y_SIZE 0x%02x \n" % y_size)
        filedh.write("__attribute__((aligned(16))) volatile uint32_t y_dump[STIM_Y_SIZE] = {\n")

        for i in range(self.H_OUT_int):
            for k_out_major in range(self.N_TILES_K_OUT):
                for qao_major in range(self.N_TILES_QA_OUT):
                    for b in range(self.HW_BC_BLOCK_SIZE):
                        for j in range(self.W_OUT_int):
                            if (i<self.H_OUT and j<self.W_OUT):
                                value = 0
                                for k in range(0,self.HW_TP-1,8):
                                    bits = y_image[i,j,k_out_major,qao_major,b,k:k+8]
                                    dec = bin2dec(bits)
                                    value = value + (dec << k)
                                file.write("0x%02x,\n" % value)
                                filedh.write("0x%02x,\n" % 0) #write zero
                            else:
                                value = ((i+1)*(j+1)) & 0xff;
                                file.write("0x%02x,\n" % value)
                                filedh.write("0x%02x,\n" % value) #write zero

        file.write("}; \n")
        filedh.write("}; \n")
        file.close()
        #filed.close()
        filedh.close()


    def create_x_image(self):
        """
            Bring input activations in the form needed to load into memory for the RBE accelerator
        """
        print("""    ### RBE - print x image """)
        x_image = np.zeros(shape=(self.H_IN_int, self.W_IN_int, self.N_TILES_K_IN, self.N_TILES_QA, self.HW_BC_BLOCK_SIZE, self.HW_TP), dtype=np.int64)

        for i in range(self.H_IN_int):
            for j in range(self.W_IN_int):
                for k_in_major in range(self.N_TILES_K_IN):
                    for qa_major in range(self.N_TILES_QA):
                        for qa_minor in range(self.HW_BC_BLOCK_SIZE if qa_major<(self.QA//self.HW_BC_BLOCK_SIZE) else self.QA_REST):
                            qai = qa_major*self.HW_BC_BLOCK_SIZE + qa_minor
                            for k_in_minor in range(self.HW_TP if k_in_major<(self.K_IN//self.HW_TP) else self.K_IN_REST):
                                k_in = self.HW_TP*k_in_major + k_in_minor
                                activ  = get_index(self.xabs[i, j, k_in], qai)
                                x_image[i, j, k_in_major, qa_major, qa_minor, k_in_minor] = activ
        self.x_image = x_image


    def create_y_image(self):
        """
            Bring output activations in the form needed to compare the RBE accelerator result with it
        """
        print("""    ### RBE - create y images """)
        y_image = np.zeros(shape=(self.H_OUT_int, self.W_OUT_int, self.N_TILES_K_OUT, self.N_TILES_QA_OUT, self.HW_BC_BLOCK_SIZE, self.HW_TP), dtype=np.int64)

        for i in range(self.H_OUT_int):
            for j in range(self.W_OUT_int):

                for k_out_major in range(self.N_TILES_K_OUT):

                    for qa_major in range(self.N_TILES_QA_OUT):
                        for qa_minor in range(self.HW_BC_BLOCK_SIZE if qa_major<(self.QA_OUT//self.HW_BC_BLOCK_SIZE) else self.QA_OUT_REST):

                            qai = qa_major*self.HW_BC_BLOCK_SIZE + qa_minor
                            # print(qai)

                            for k_out_minor in range(self.HW_TP if k_out_major<(self.K_OUT//self.HW_TP) else self.K_OUT_REST):

                                k_out = self.HW_TP*k_out_major + k_out_minor

                                output  = get_index(self.y_bittrue[i, j, k_out], qai)
                                y_image[i,j, k_out_major, qa_major, qa_minor, k_out_minor] = output

        self.y_image = y_image


    def create_x_patches(self):
        """
            Create an patchs of the activation that fits into the Activation buffer of RBE
        """
        print("""    ### RBE - create x patches """)

        # initialize
        x_patch = np.zeros(shape=(self.N_XPATCHES, self.F_BUFFER_SIZE, self.F_BUFFER_SIZE, self.HW_BC_BLOCK_SIZE, self.HW_TP), dtype=np.int64)

        # filtersize 1
        if self.FS == 1:

            for i_major in range(self.N_TILES_H_OUT):
                for j_major in range(self.N_TILES_W_OUT):

                    for i_minor in range(int(np.sqrt(self.HW_NR_COLUMN)) if i_major<(self.H_IN_int//np.sqrt(self.HW_NR_COLUMN)) else self.H_IN_int_REST):
                        for j_minor in range(int(np.sqrt(self.HW_NR_COLUMN)) if j_major<(self.W_IN_int//np.sqrt(self.HW_NR_COLUMN)) else self.W_IN_int_REST):
                            i = int(i_major*3 + i_minor)
                            j = int(j_major*3 + j_minor)

                            for k_in_major in range(self.N_TILES_K_IN):

                                for qa_major in range(self.N_TILES_QA):
                                    for qa_minor in range(self.HW_BC_BLOCK_SIZE if qa_major<(self.QA//self.HW_BC_BLOCK_SIZE) else self.QA_REST):
                                        qai = qa_major*self.HW_BC_BLOCK_SIZE + qa_minor

                                        for k_in_minor in range(self.HW_TP if k_in_major<(self.K_IN//self.HW_TP) else self.K_IN_REST):

                                            k_in = self.HW_TP*k_in_major + k_in_minor
                                            activ  = get_index(self.xabs[i, j, k_in], qai)
                                            x_patch[i_major*self.N_TILES_W_OUT*self.N_TILES_QA*self.N_TILES_K_IN + j_major*self.N_TILES_QA*self.N_TILES_K_IN + k_in_major*self.N_TILES_QA+qa_major, i_minor, j_minor, qa_minor, k_in_minor] = activ

            self.x_patch = x_patch
            x_flipped = np.flip(np.flip(np.flip(np.flip(x_patch, 3), 2), 1), 4)
            self.x_patch_save = np.reshape(x_flipped, newshape=(self.N_XPATCHES, self.F_BUFFER_SIZE*self.F_BUFFER_SIZE*self.HW_BC_BLOCK_SIZE*self.HW_TP))
            # np.savetxt(self.dir_golden_bittrue+'x_patch_save.txt', self.x_patch_save, fmt='%01.1x', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)

        # filtersize 3
        elif self.FS == 3:

            for i_major in range(self.N_TILES_H_OUT):
                for j_major in range(self.N_TILES_W_OUT):

                    for i_minor in range(self.F_BUFFER_SIZE): # if i_major<(self.H_IN_int//self.F_BUFFER_SIZE) else self.H_IN_int_REST):
                        for j_minor in range(self.F_BUFFER_SIZE): # if j_major<(self.W_IN_int//self.F_BUFFER_SIZE) else self.W_IN_int_REST):
                            i = int(i_major*(self.F_BUFFER_SIZE-2) + i_minor)
                            j = int(j_major*(self.F_BUFFER_SIZE-2) + j_minor)

                            for k_in_major in range(self.N_TILES_K_IN):

                                for qa_major in range(self.N_TILES_QA):
                                    for qa_minor in range(self.HW_BC_BLOCK_SIZE if qa_major<(self.QA//self.HW_BC_BLOCK_SIZE) else self.QA_REST):
                                        qai = qa_major*self.HW_BC_BLOCK_SIZE + qa_minor

                                        for k_in_minor in range(self.HW_TP if k_in_major<(self.K_IN//self.HW_TP) else self.K_IN_REST):

                                            k_in = self.HW_TP*k_in_major + k_in_minor

                                            activ  = get_index(self.xabs[i, j, k_in], qai)
                                            x_patch[i_major*self.N_TILES_W_OUT*self.N_TILES_QA*self.N_TILES_K_IN + j_major*self.N_TILES_QA*self.N_TILES_K_IN + k_in_major*self.N_TILES_QA+qa_major, i_minor, j_minor, qa_minor, k_in_minor] = activ

            self.x_patch = x_patch
            x_flipped = np.flip(np.flip(np.flip(np.flip(x_patch, 3), 2), 1), 4)
            self.x_patch_save = np.reshape(x_flipped, newshape=(self.N_XPATCHES, self.F_BUFFER_SIZE*self.F_BUFFER_SIZE*self.HW_BC_BLOCK_SIZE*self.HW_TP))
            # np.savetxt(self.dir_golden_bittrue+'x_patch_save.txt', self.x_patch_save, fmt='%01.1x', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)


    def create_W_streams(self):
        print("""    ### RBE - create W streams """)

        # if filter size is 1
        if self.FS == 1:

            n_W_in_col = 1
            self.n_W_in_col = n_W_in_col
            self.N_STREAMS_PER_XPATCH = self.HW_TP+2
            print("""    ### RBE - N_STREAMS_PER_XPATCH """, self.N_STREAMS_PER_XPATCH)
            print("""    ### RBE - n_W_in_col """, self.n_W_in_col)

            W_streams = np.zeros(shape=(self.N_TILES_K_OUT, self.N_TILES_K_IN, self.N_STREAMS_PER_XPATCH, self.HW_BC_COLUMN_SIZE, self.HW_TP), dtype=np.int64)

            W_streams[:,:,0, 0, :] = 1
            W_streams[:,:,1, 0, :] = 1

            for k_out_major in range(self.N_TILES_K_OUT):
                for k_in_major in range(0, self.N_TILES_K_IN-n_W_in_col+1, n_W_in_col):
                    for k_out_minor in range(self.HW_TP if k_out_major<(self.K_OUT//self.HW_TP) else self.K_OUT_REST):
                        k_out = self.HW_TP*k_out_major + k_out_minor

                        for r in range(self.QW):
                            wi = r // self.QW
                            qwi = (r - wi*self.QW) % self.QW

                            if wi >= n_W_in_col:
                                break

                            for k_in_minor in range(self.HW_TP if k_in_major<(self.K_IN//self.HW_TP) else self.K_IN_REST):
                                k_in = self.HW_TP*(k_in_major+wi) + k_in_minor

                                weight = get_index(self.Wsabs[k_out, 0, 0, k_in], qwi)
                                W_streams[k_out_major, k_in_major, 2+k_out_minor, r, k_in_minor] = weight # TODO k_out_minor correct?

            self.W_streams = W_streams
            W_flipped = np.flip(np.flip(W_streams[0,0], 1), 2)
            # self.W_streams_save = np.reshape(W_flipped, newshape=(self.N_TILES_K_OUT*self.N_TILES_K_IN*self.N_STREAMS_PER_XPATCH, self.HW_BC_COLUMN_SIZE * self.HW_TP))
            # np.savetxt(self.dir_golden_bittrue+'W_streams_save.txt', self.W_streams_save, fmt='%01.1x', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)

        # if filter size is 3
        elif self.FS==3:

            self.N_STREAMS_PER_XPATCH = self.HW_TP*(self.QW)+2
            print("""    ### RBE - N_STREAMS_PER_XPATCH """, self.N_STREAMS_PER_XPATCH)

            W_streams = np.zeros(shape=(self.N_TILES_K_OUT, self.N_TILES_K_IN, self.N_STREAMS_PER_XPATCH, self.HW_BC_COLUMN_SIZE, self.HW_TP), dtype=np.int64)

            W_streams[:,:,0, :, :] = 1
            W_streams[:,:,1, :, :] = 1

            for k_out_major in range(self.N_TILES_K_OUT):
                for k_in_major in range(self.N_TILES_K_IN):
                    for k_out_minor in range(self.HW_TP if k_out_major<(self.K_OUT//self.HW_TP) else self.K_OUT_REST):
                        k_out = self.HW_TP*k_out_major + k_out_minor

                        # one loop more for the signed_offset
                        # for qwi in range(self.QW,-1,-1):
                        for qwi in range(self.QW):

                            for r in range(self.HW_BC_COLUMN_SIZE):
                                fi = r // self.FS
                                fj = r % self.FS

                                for k_in_minor in range(self.HW_TP if k_in_major<(self.K_IN//self.HW_TP) else self.K_IN_REST):
                                    k_in = self.HW_TP*k_in_major + k_in_minor

                                    weight = get_index(self.Wsabs[k_out, fi, fj, k_in], qwi)
                                    W_streams[k_out_major, k_in_major, 2+k_out_minor*(self.QW) + qwi, r, k_in_minor] = weight # TODO k_out_minor correct?

            self.W_streams = W_streams
            W_flipped = np.flip(np.flip(W_streams[0,0], 1), 2)
            # self.W_streams_save = np.reshape(W_flipped, newshape=(self.N_TILES_K_OUT*self.N_TILES_K_IN*self.N_STREAMS_PER_XPATCH, self.HW_BC_COLUMN_SIZE * self.HW_TP))
            # np.savetxt(self.dir_golden_bittrue+'W_streams_save.txt', self.W_streams_save, fmt='%01.1x', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)


    def create_y_patches(self):

        print("""    ### RBE - create y patches """)
        y_patch = np.zeros(shape=(self.N_YPATCHES, 3, 3, self.HW_BC_BLOCK_SIZE, self.HW_TP), dtype=np.int64)

        for i_major in range(self.N_TILES_H_OUT):
            for j_major in range(self.N_TILES_W_OUT):

                for i_minor in range(int(np.sqrt(self.HW_NR_COLUMN)) if i_major<(self.H_OUT_int//np.sqrt(self.HW_NR_COLUMN)) else self.H_OUT_REST):
                    for j_minor in range(int(np.sqrt(self.HW_NR_COLUMN)) if j_major<(self.W_OUT_int//np.sqrt(self.HW_NR_COLUMN)) else self.W_OUT_REST):

                        i = int(i_major * 3 + i_minor)
                        j = int(j_major * 3 + j_minor)
                        # print(i,j)

                        for k_out_major in range(self.N_TILES_K_OUT):

                            # for qa_major in range(self.N_TILES_QA-1,-1,-1):
                            for qa_major in range(self.N_TILES_QA_OUT):
                                for qa_minor in range(self.HW_BC_BLOCK_SIZE if qa_major<(self.QA_OUT//self.HW_BC_BLOCK_SIZE) else self.QA_OUT_REST):

                                    qai = qa_major*self.HW_BC_BLOCK_SIZE + qa_minor
                                    # print(qai)

                                    for k_out_minor in range(self.HW_TP if k_out_major<(self.K_OUT//self.HW_TP) else self.K_OUT_REST):

                                        k_out = self.HW_TP*k_out_major + k_out_minor

                                        output  = get_index(self.y_bittrue[i, j, k_out], qai)
                                        y_patch[i_major*self.N_TILES_W_OUT*self.N_TILES_QA_OUT*self.N_TILES_K_OUT + j_major*self.N_TILES_QA_OUT*self.N_TILES_K_OUT + k_out_major*self.N_TILES_QA_OUT+qa_major, i_minor, j_minor, qa_minor, k_out_minor] = output

        # print(x_patch)
        self.y_patch = y_patch
        y_flipped = np.flip(np.flip(np.flip(np.flip(y_patch, 3), 2), 1), 4)
        self.y_size = np.reshape(y_flipped, newshape=(self.N_YPATCHES, 3*3*self.HW_BC_BLOCK_SIZE*self.HW_TP))

        np.savetxt(self.dir_golden_bittrue+'y_patch_save.txt', self.x_patch_save, fmt='%01.1x', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)


    # TODO: currently hardcoded to fs=3
    def extract_x_array(self, idx=0):
        print("""    ### RBE - extract x array id={}/{}""".format(idx, self.N_XPATCHES))

        # pixel layout FS==1
        #
        #  0  1  2  -  -
        #  5  6  7  -  -
        # 10 11 12  -  -
        #  -  -  -  -  -
        #  -  -  -  -  -
        if self.FS == 1:
            x_array = np.zeros(shape=(self.HW_NR_COLUMN, self.HW_BC_COLUMN_SIZE, self.HW_BC_BLOCK_SIZE, self.HW_TP), dtype=np.int64)

            for c in range(self.HW_NR_COLUMN):
                for r in range(self.HW_BC_COLUMN_SIZE):

                    i = int(c // np.sqrt(self.HW_NR_COLUMN))
                    j = int(c % np.sqrt(self.HW_NR_COLUMN))

                    x_array[c,r,:,:] = self.x_patch[idx, i, j, :, :]

        # pixel layout FS==3
        #
        #  0  1  2  3  4
        #  5  6  7  8  9
        # 10 11 12 13 14
        # 15 16 17 18 19
        # 20 21 22 23 24
        elif self.FS == 3:
            x_array = np.zeros(shape=(self.HW_NR_COLUMN, self.HW_BC_COLUMN_SIZE, self.HW_BC_BLOCK_SIZE, self.HW_TP), dtype=np.int64)

            for c in range(self.HW_NR_COLUMN):
                for r in range(self.HW_BC_COLUMN_SIZE):

                    fi = r // self.FS
                    fj = r % self.FS

                    i = c // self.FS + fi
                    j = c % self.FS + fj

                    x_array[c,r,:,:] = self.x_patch[idx, i, j, :, :]

        # save the x_array
        x_array_save = np.reshape(x_array, newshape=(self.HW_NR_COLUMN*self.HW_BC_COLUMN_SIZE, self.HW_BC_BLOCK_SIZE*self.HW_TP))
        np.savetxt(self.dir_golden_bittrue+'x_array_save.txt', x_array_save, fmt='%01.1x', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)

        return x_array


    def run_simple_golden_network(self, print_y=True):
        """
        runs a simple golden model with the standard quantization with
        non-shifted weigts (if generated signed)

        Parameters
        ----------
        print_y : Boolean (default=True)
            True: write the results into a file
        """
        print("""    ### RBE - START  - Simple Golden Model """)

        y = np.zeros(shape=(self.H_OUT_int, self.W_OUT_int,
                            self.K_OUT_int), dtype=np.int64)

        # Initialize lists to trace and dump to file
        outputs = []
        outputs_nonqnt = []

        i = 0
        j = 0
        k_out_major = 0
        fi = 0
        fj = 0
        k_in_major = 0
        k_out_minor = 0
        k_in_minor = 0

        # just to the effective rows and columns
        for i in range(self.H_OUT): # take real size
            for j in range(self.W_OUT): # take real size

                for k_out_major in range(self.N_TILES_K_OUT):

                    for fi in range(self.FS):
                        for fj in range(self.FS):

                            for k_in_major in range(self.N_TILES_K_IN):
                                for k_out_minor in range(self.HW_TP if k_out_major < (self.K_OUT // self.HW_TP) else self.K_OUT_REST):

                                    k_out = self.HW_TP * k_out_major + k_out_minor

                                    for k_in_minor in range(self.HW_TP if k_in_major < (self.K_IN // self.HW_TP) else self.K_IN_REST):
                                        k_out = self.HW_TP * k_out_major + k_out_minor
                                        k_in = self.HW_TP * k_in_major + k_in_minor
                                        if self.QW == 1:
                                            weight = self.W[k_out, fi, fj, k_in]*2-1
                                        else:
                                            weight = self.W[k_out, fi, fj, k_in]
                                        if self.QA == 1:
                                            activ = self.x[i + fi, j + fj, k_in]*2-1
                                        else:
                                            activ = self.x[i + fi, j + fj, k_in]

                                        # multiply with floor replicates right shift
                                        mult = int(math.floor(weight * activ))

                                        y[i, j, k_out] = add_saturate(y[i, j, k_out],
                                                                      mult,
                                                                      self.BINCONV_ACCUM_UPPER_BOUND,
                                                                      self.BINCONV_ACCUM_LOWER_BOUND)
                                        #if (i==self.H_OUT_int-1 and j==self.W_OUT_int-1):
                                        #    print("last value of channel "+str(k_out)+"is "+str(mult)+" "+str(y[i, j, k_out]))

        # Batch normalization
        for i in range(self.H_OUT_int):
            for j in range(self.W_OUT_int):
                for k_out_major in range(self.N_TILES_K_OUT):

                    # Base shift + quantize + ReLU
                    for k_out_minor in range(self.HW_TP if k_out_major < (self.K_OUT // self.HW_TP) else self.K_OUT_REST):
                        k_out = self.HW_TP * k_out_major + k_out_minor
                        output = y[i, j, k_out]
                        outputs_nonqnt.append(y[i, j, k_out])
                        if (i<self.H_OUT and j<self.W_OUT):
                            if self.output_quant:
                                y[i, j, k_out] = ((int(math.floor(output)) * self.kappa_bn[k_out] + self.lambda_bn[k_out]) >> self.shift_reqnt)
                                if self.RELU:
                                    y[i, j, k_out] = y[i, j, k_out].clip(0, 2**self.QA_OUT-1)
                                else:
                                    y[i, j, k_out] = y[i, j, k_out].clip(-2**(self.QA_OUT-1), 2**(self.QA_OUT-1)-1)
                            else:
                                y[i, j, k_out] = (int(math.floor(output)))
                        outputs.append(y[i, j, k_out])

        self.y_golden = np.transpose(y, (2,0,1))

        # write results to file
        if print_y:
            dump_to_file(['{0:x}'.format(elem) for elem in outputs],
                         self.dir_golden_simple + 'y.txt')

        if print_y:
            dump_to_file(['{0:x}'.format(elem) for elem in outputs_nonqnt],
                         self.dir_golden_simple + 'y_nonqnt.txt')

        print("""    ### RBE - FINISH - Simple Golden Model """)
        self.print_hashline()


    def run_simple_shifted_network(self, print_y=True, verify_model=True):
        """
        Runs a simple golden model with the so-called offset computation
        with the standard quantization with non-shifted weigts (if generated signed)
        In a first step, all input channels

        Parameters
        ----------
        print_y : Boolean (default=True)
            True: write the results into a file
        verify_model : Boolean (default=True)
            Compare the model against the simple golden model
        """
        self.print_hashline()
        print("""    ### RBE - START  - Simple Shifted Model """)


        y = np.zeros(shape=(self.H_OUT_int, self.W_OUT,
                            self.K_OUT_int), dtype=np.int64)
        act_sum = np.zeros(shape=(self.H_OUT_int, self.W_OUT_int), dtype=np.int64)

        # Initialize lists to trace and dump to file
        outputs = []

        i = 0
        j = 0
        k_out_major = 0
        fi = 0
        fj = 0
        k_in_major = 0
        k_out_minor = 0
        k_in_minor = 0

        # compute offsets
        for i in range(self.H_OUT_int):     # real size
            for j in range(self.W_OUT_int): # real size

                for fi in range(self.FS):
                    for fj in range(self.FS):

                        for k_in_major in range(self.N_TILES_K_IN):
                            for k_in_minor in range(self.HW_TP if k_in_major < (self.K_IN // self.HW_TP) else self.K_IN_REST):

                                k_in = self.HW_TP * k_in_major + k_in_minor
                                act = int(self.x[i + fi, j + fj, k_in])
                                act_sum[i, j] = add_saturate(
                                    act_sum[i, j], act, self.BINCONV_ACCUM_UPPER_BOUND, self.BINCONV_ACCUM_LOWER_BOUND)

        act_sum = np.multiply(act_sum, -self.Wmin)

        for i in range(self.H_OUT_int):     #real size
            for j in range(self.W_OUT_int): #real size

                for k_out_major in range(self.N_TILES_K_OUT):

                    for fi in range(self.FS):
                        for fj in range(self.FS):

                            for k_in_major in range(self.N_TILES_K_IN):

                                for k_out_minor in range(self.HW_TP if k_out_major < (self.K_OUT // self.HW_TP) else self.K_OUT_REST):

                                    for k_in_minor in range(self.HW_TP if k_in_major < (self.K_IN // self.HW_TP) else self.K_IN_REST):
                                        k_out = self.HW_TP * k_out_major + k_out_minor
                                        k_in = self.HW_TP * k_in_major + k_in_minor

                                        weight = self.Wshifted[k_out, fi, fj, k_in]
                                        activ = self.x[i + fi, j + fj, k_in]

                                        # multiply with floor replicates right shift
                                        mult = int(math.floor(weight * activ))

                                        y[i, j, k_out] = add_saturate(
                                            y[i, j, k_out], mult, self.BINCONV_ACCUM_UPPER_BOUND, 0)

                    for k_out_minor in range(self.HW_TP if k_out_major < (self.K_OUT // self.HW_TP) else self.K_OUT_REST):

                        k_out = self.HW_TP * k_out_major + k_out_minor
                        output = y[i, j, k_out]

                        if self.output_quant:
                            if self.RELU:
                                y[i, j, k_out] = (((int(math.floor(output)) - act_sum[i, j]) * self.kappa_bn[k_out] + self.lambda_bn[k_out]) >> self.shift_reqnt).clip(0, 2**self.QA_OUT-1)
                            else:
                                y[i, j, k_out] = (((int(math.floor(output)) - act_sum[i, j]) * self.kappa_bn[k_out] + self.lambda_bn[k_out]) >> self.shift_reqnt).clip(-2**(self.QA_OUT-1),2**(self.QA_OUT-1)-1)
                        else:
                            y[i, j, k_out] = (int(math.floor(output)) - act_sum[i, j])
                        outputs.append(y[i, j, k_out])

        self.y_simple_shifted = y

        if print_y:
            dump_to_file(['{0:x}'.format(elem) for elem in outputs],
                         self.dir_golden_simple_shifted + 'y.txt')

        print("""    ### RBE - FINISH - Simple Shifted Model """)
        self.print_hashline()

        if verify_model:
            self.print_hashline()
            print("""    ### VERIFICATION """)
            # run verification model
            self.run_simple_golden_network()
            compare_models(self.dir_golden_simple,
                           self.dir_golden_simple_shifted, self.dir_compare)


    def run_bittrue_shifted_network(self, verify_model=True):
        """
        Acutal Bittrue Golden model of the RBE

        Parameters
        ----------
        verify_model : Boolean (default=True)
            Compare the model against the simple golden model
        """
        self.print_hashline()
        print("""    ### RBE - START  - Bittrue HW Model Optimized Offset Shift """)

        # Initialize lists to trace and dump to file
        outputs         = []
        outputs_nonqnt  = []

        # generate x patches
        self.create_x_patches()
        self.create_x_image()
        # generate weight streams
        self.create_W_streams()

        # write out W and x stimuli for accelerator verification
        self.write_file_x_image()
        self.write_file_w_streams()

        # initialization
        popcount = 0
        comb     = np.zeros(shape=(self.N_XPATCHES, self.N_TILES_K_OUT, self.HW_NR_COLUMN, self.HW_BC_COLUMN_SIZE, self.HW_TP), dtype=np.int64)
        y        = np.zeros(shape=(self.N_XPATCHES, self.N_TILES_K_OUT, self.HW_NR_COLUMN, self.HW_TP), dtype=np.int64)
        o        = np.zeros(shape=(self.H_OUT_int, self.W_OUT_int, self.K_OUT_int), dtype=np.int64)
        yb       = np.zeros(shape=(self.K_OUT, self.H_OUT_int*self.W_OUT_int), dtype='object')

        ##
        ## compute on HW mapping
        ##
        for k_out_major in range(self.N_TILES_K_OUT):
            for p in range(self.N_XPATCHES):

                qa_major = p % self.N_TILES_QA

                k_in_major = (int(((p-qa_major))/(self.N_TILES_QA)))% self.N_TILES_K_IN

                i_major = int(math.ceil( (p - k_in_major*self.N_TILES_QA - qa_major) // (self.N_TILES_W_OUT*self.N_TILES_QA*self.N_TILES_K_IN) ))
                j_major = int(math.ceil( ((p - i_major*self.N_TILES_W_OUT*self.N_TILES_QA*self.N_TILES_K_IN - k_in_major*self.N_TILES_QA - qa_major)//(self.N_TILES_QA*self.N_TILES_K_IN)) ))


                x_array = self.extract_x_array(p)
                # print(x_array)
                for s in range(self.N_STREAMS_PER_XPATCH):
                    # print(s)

                    ##
                    ## compute offset
                    ##
                    if s==0 or s==1:
                        for c in range(self.HW_NR_COLUMN):
                            # pixel_add
                            for r in range(self.HW_BC_COLUMN_SIZE):

                                if self.FS == 1:

                                    if (r % self.QW) > 0 or r//self.QW > self.n_W_in_col-1:
                                        break

                                    for b in range(self.HW_BC_BLOCK_SIZE):

                                        # popcount computation in every BinConv
                                        for t in range(self.HW_TP):

                                            weight = self.W_streams[k_out_major,k_in_major,s, r, t]
                                            activ  = x_array[c, r, b, t]

                                            mult = weight & activ

                                            popcount = add_saturate(popcount, mult, self.BINCONV_ACCUM_UPPER_BOUND, self.BINCONV_ACCUM_LOWER_BOUND)



                                        # scale and combination tree after accumulate
                                        if s==0:
                                            scaled_tmp = int(popcount*int(get_scale(b+qa_major*self.HW_BC_BLOCK_SIZE,(self.QW-1)))) # TODO: in HW "self.QW-1" should be independent from QW
                                            # print(scaled_tmp)
                                            # add scaeld value to results
                                            comb[p, k_out_major, c, r, :] = vadd_saturate(comb[p, k_out_major, c, r, :], -scaled_tmp, self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)
                                            # print(comb[p, c, r, :])
                                        else:
                                            # scale the tmp result
                                            scaled_tmp = int(popcount*int(get_scale(b+qa_major*self.HW_BC_BLOCK_SIZE, 0)))

                                            # add scaeld value to results
                                            comb[p, k_out_major, c, r, :] = vadd_saturate(comb[p, k_out_major, c, r, :], scaled_tmp, self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)

                                        popcount = 0
                                    y[p, k_out_major, c, :] = vadd_saturate(y[p, k_out_major, c, :], comb[p, k_out_major, c, r, :], self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)
                                    comb[p, k_out_major, c, r, :] = 0


                                elif self.FS == 3:

                                    # popcount, scale and combination tree
                                    # for b in range(self.HW_BC_BLOCK_SIZE-1,-1,-1):
                                    for b in range(self.HW_BC_BLOCK_SIZE):

                                        # popcount computatio in every BinConv
                                        for t in range(self.HW_TP):

                                            weight = self.W_streams[k_out_major,k_in_major,s, r, t]
                                            activ  = x_array[c, r, b, t]

                                            mult = weight & activ

                                            # print(weight, activ, mult)
                                            popcount = add_saturate(popcount, mult, self.BINCONV_ACCUM_UPPER_BOUND, self.BINCONV_ACCUM_LOWER_BOUND)
                                        # print(popcount)


                                        # scale and combination tree after accumulate
                                        if s==0:
                                            scaled_tmp = int(popcount*int(get_scale(b+qa_major*self.HW_BC_BLOCK_SIZE,(self.QW-1)))) # TODO: in HW "self.QW-1" should be independent from QW
                                            # if c==0:
                                                # print(-scaled_tmp)
                                            # add scaeld value to results
                                            comb[p, k_out_major, c, r, :] = vadd_saturate(comb[p, k_out_major, c, r, :], -scaled_tmp, self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)
                                            # print(comb[p, c, r, :])
                                            # if c==0:
                                                # print(comb[p, k_out_major, c, r, :])
                                        else:
                                            # scale the tmp result
                                            scaled_tmp = int(popcount*int(get_scale(b+qa_major*self.HW_BC_BLOCK_SIZE, 0)))
                                            # if c==0:
                                                # print(scaled_tmp)
                                            # add scaeld value to results
                                            comb[p, k_out_major, c, r, :] = vadd_saturate(comb[p, k_out_major, c, r, :], scaled_tmp, self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)
                                            # if c==0:
                                                # print(comb[p, k_out_major, c, r, :])
                                        popcount = 0
                                        # print(comb[p, c, r, k_out])
                                    # print(comb[p, k_out_major, c, r, 0])
                                    y[p, k_out_major, c, :] = vadd_saturate(y[p, k_out_major, c, :], comb[p, k_out_major, c, r, :], self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)
                                    comb[p, k_out_major, c, r, :] = 0

                    # no special shift offset computation
                    else:

                        k_out = 0
                        qw = 0

                        if self.FS==1:
                            k_out = (s-2) % (self.N_STREAMS_PER_XPATCH)

                        elif self.FS==3:
                            qw    = (s-2) % (self.QW)
                            k_out = (s-2) // (self.QW)
                        # print(qw, k_out)

                        for c in range(self.HW_NR_COLUMN):
                            # pixel_add
                            for r in range(self.HW_BC_COLUMN_SIZE):

                                if self.FS==1:
                                    qw = r % self.QW
                                # print("qw: ", qw)
                                # popcount, scale and combination tree
                                # for b in range(self.HW_BC_BLOCK_SIZE-1,-1,-1):
                                for b in range(self.HW_BC_BLOCK_SIZE):

                                    # popcount computatio in every BinConv
                                    for t in range(self.HW_TP):

                                        weight = self.W_streams[k_out_major,k_in_major,s, r, t]
                                        activ  = x_array[c, r, b, t]

                                        mult = weight & activ

                                        # print(weights, activ, mult)
                                        popcount = add_saturate(popcount, mult, self.BINCONV_ACCUM_UPPER_BOUND, self.BINCONV_ACCUM_LOWER_BOUND)
                                    # print(popcount)

                                    scaled_tmp = int(popcount*int(get_scale(b+qa_major*self.HW_BC_BLOCK_SIZE, qw)))
                                    # print(scaled_tmp)
                                    # add scaeld value to results
                                    comb[p, k_out_major, c, r, k_out] = add_saturate(comb[p, k_out_major, c, r, k_out], scaled_tmp, self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)

                                    # if c==0:
                                    #     print(comb[p, c, 0, 0])
                                    popcount = 0
                                    # print(comb[p, c, r, k_out])

                                y[p, k_out_major, c, k_out] = add_saturate(y[p, k_out_major, c, k_out], comb[p, k_out_major, c, r, k_out], self.COMBINATION_ACCUM_UPPER_BOUND, self.COMBINATION_ACCUM_LOWER_BOUND)
                                # if c==0:
                                #     print(comb[p, k_out_major, c, r, k_out])
                                comb[p, k_out_major, c, r, k_out] = 0

                for c in range(self.HW_NR_COLUMN):
                    if self.FS==1:
                        i = int(c // np.sqrt(self.HW_NR_COLUMN)) + i_major*3 #+ (p * int(self.N_TILES_H_OUT/self.N_TILES_K_IN))
                        j = int(c  % np.sqrt(self.HW_NR_COLUMN)) + j_major*3 #+ (p * int(self.N_TILES_W_OUT/self.N_TILES_K_IN))

                        o[i, j, (k_out_major)*self.HW_TP:(k_out_major+1)*self.HW_TP] += y[p, k_out_major, c, :]
                    elif self.FS==3:
                        i = (c // self.FS) + i_major*3 #(int(p/(self.N_TILES_K_IN*self.N_TILES_W_OUT*self.N_TILES_QA)))
                        j = (c  % self.FS) + j_major*3 #(int(p*3/(self.N_TILES_K_IN*self.N_TILES_H_OUT*self.N_TILES_QA)))

                        if (i<self.H_OUT and j<self.W_OUT):
                            o[i, j, (k_out_major)*self.HW_TP:(k_out_major+1)*self.HW_TP] += y[p, k_out_major, c, :]
        clip_min = 0 if self.RELU else -2**(self.QA_OUT-1);
        clip_max = 2**(self.QA_OUT)-1 if self.RELU else 2**(self.QA_OUT-1)-1;

        for i_major in range(self.N_TILES_H_OUT):
            for i_minor in range(int(self.HW_NR_COLUMN/3) if i_major<(self.H_OUT_int//(self.HW_NR_COLUMN/3)) else self.H_OUT_REST):
                for j_major in range(self.N_TILES_W_OUT):
                    for j_minor in range(int(self.HW_NR_COLUMN/3) if j_major<(self.W_OUT_int//(self.HW_NR_COLUMN/3)) else self.W_OUT_REST):

                        i = int(i_major*(self.HW_NR_COLUMN/3) + i_minor)
                        j = int(j_major*(self.HW_NR_COLUMN/3) + j_minor)


                        for k_out_major in range(self.N_TILES_K_OUT):
                            for k_out_minor in range(self.HW_TP if k_out_major<(self.K_OUT//self.HW_TP) else self.K_OUT_REST):

                                k_out = self.HW_TP*k_out_major + k_out_minor
                                # norm-requant + writing out results
                                output = o[i, j, k_out]
                                #silence of padded rows and cols
                                if (i<self.H_OUT and j<self.W_OUT):
                                    if self.output_quant:
                                        o[i, j, k_out]  = ((int(math.floor(output)) * self.kappa_bn[k_out] + self.lambda_bn[k_out]) >> self.shift_reqnt).clip(clip_min, clip_max)

                                    else:
                                        o[i, j, k_out]  = (int(math.floor(output)))
                                outputs_nonqnt.append(output)
                                outputs.append(o[i, j, k_out])


        self.y_bittrue = o

        self.create_y_patches()
        self.create_y_image()
        self.write_file_y_image()

        y_binary_out =  np.reshape(np.transpose(o, (2,0,1)), newshape=(self.K_OUT_int, self.H_OUT_int*self.W_OUT_int))
        # Silence non-relevant cols and rows
        self.y_bittrue[:,:,self.K_OUT:self.K_OUT_int].fill(0)

        for a in range(self.K_OUT):
            for b in range(self.H_OUT_int*self.W_OUT_int):
                yb[a, b] = (np.binary_repr(y_binary_out[a, b], width=32)) #.zfill(16)


        np.savetxt(self.dir_golden_bittrue+'y_binary.txt', yb, fmt='%s', delimiter='', newline='\n', header='', footer='', comments='# ', encoding=None)

        # write results to file
        dump_to_file(['{0:x}'.format(elem)for elem in outputs_nonqnt], self.dir_golden_bittrue+'y_nonqnt.txt')

        dump_to_file(['{0:x}'.format(elem)for elem in outputs], self.dir_golden_bittrue+'y.txt')

        dump_to_file(['{0:d}'.format(elem)for elem in outputs], self.dir_golden_bittrue+'y_comp.txt')

        print("""    ### RBE - FINISH - Bittrue Model """)
        self.print_hashline()

        if verify_model:
            self.print_hashline()
            print("""    ### VERIFICATION with simple golden model""")
            # run verification model
            self.run_simple_golden_network()
            compare_models(self.dir_golden_simple,
                           self.dir_golden_bittrue, self.dir_compare)
            self.print_hashline()
            print("""    ### check non-quantized outputs""")
            compare_models(self.dir_golden_simple,
                           self.dir_golden_bittrue, self.dir_compare, compare_file='y_nonqnt.txt')
