#!/usr/bin/env python3.6
#
# helper_functions.py
#
# Gianna Paulin <pauling@iis.ee.ethz.ch>
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

def compare(golden_file, actual_file, result_file):
    """
        Compare the contents of two files.

        Parameters
        ----------
        golden_file : String
            Filename of the file containing the golden results
        actual_file : String
            Filename of the file containing the results to be checked
        result_file : String
            Filename of the file in which the comparison results are written to
    """
    f1 = open(golden_file, 'r')
    f2 = open(actual_file, 'r')
    f3 = open(result_file, 'w+')

    # Read in all the lines of your file into a list of lines
    lines1 = f1.readlines()
    lines2 = f2.readlines()

    # check the length of the lines of both files
    if len(lines1) > len(lines2):
        print('Warning!\tFile {} has {} elements. More than {}\'s {} elements'.format(
            golden_file, len(lines1), actual_file, len(lines2)))
    elif len(lines2) > len(lines1):
        print('Warning!\tFile {} has {} elements. More than {}\'s {} elements'.format(
            actual_file, len(lines2), golden_file, len(lines1)))

    errors = 0
    total = 0
    for idx, elem1 in enumerate(lines1):
        if len(lines2) > idx:
            elem2 = lines2[idx]
            if elem2 != elem1:
                f3.write('Error!\tGolden: {}\tActual: {}\n'.format(elem1, elem2))
                errors += 1
            else:
                f3.write('OK!\tGolden: {}\tActual: {}\n'.format(elem1, elem2))
            total += 1
        else:
            f3.write('Error!\tGolden: {}\tActual: out of bounds\n'.format(elem1))
    f3.write('\nTotal Errors: {}/{}'.format(errors, total))
    f1.close()
    f2.close()
    f3.close()
    return errors, total


def dump_to_file(list, filename):
    """
        Writes all elements in `list` to file `filename`

        Parameters
        ----------
        list : Array
            Array of elements
        filename : String
            Filename into which all elements from `list` are written to
    """
    s = '\n'.join([elem for elem in list])
    f = open(filename, 'w+')
    f.write(s)
    f.close()


def compare_models_numpy(golden, actual, heatmap=False):
    """
        Compare two feature maps for equivalence.

        Parameters
        ----------
        golden : 3D numpy Tensor
            Expected Responses (Golden Model)
        actual : 3D numpy Tensor
            Actual Results
        heatmap : Boolean (default=False)
            If True a heatmap with all difference for every channels is ploted.
    """
    # Make both model equal size
    num_ch = max(np.shape(golden)[0], np.shape(actual)[0])
    height = max(np.shape(golden)[1], np.shape(actual)[1])
    width  = max(np.shape(golden)[2], np.shape(actual)[2])
    golden_reshaped = np.zeros((num_ch, height, width), dtype=np.int64)#.fill(math.nan)
    actual_reshaped = np.zeros((num_ch, height, width), dtype=np.int64)#.fill(math.nan)
    golden_reshaped[0:np.shape(golden)[0], 0:np.shape(golden)[1], 0:np.shape(golden)[2]] = golden
    actual_reshaped[0:np.shape(actual)[0], 0:np.shape(actual)[1], 0:np.shape(actual)[2]] = actual
    comparison = np.equal(golden_reshaped, actual_reshaped)

    print(str(int((comparison.sum())))+"/"+str(np.product(np.shape(comparison)))+" are correct")

    if heatmap and int(comparison.sum())!=np.product(np.shape(comparison)):
        try:
            plt,sqrt,ceil
        except NameError:
            import matplotlib.pyplot as plt
            from math import sqrt
            from math import ceil
        try:
            num_ch = comparison.shape[0]
            fig = plt.figure()
            ax_list = list()
            ratio = 1.7
            rows = sqrt(num_ch/ratio)
            cols = ceil(rows*ratio)
            rows = ceil(rows)
            for i in range(num_ch):
                # todo if all values correct, it shows all black instead all white #.astype(int)
                ax_list.append(fig.add_subplot(rows,cols,i+1)) #
                ax_list[i].imshow(comparison[i, :,:], cmap='hot', vmin=0, vmax=1)

        except:
            print("DISPLAY not available.")
        plt.show()
    #return comparison


def compare_models(golden_simple, golden_bittrue, dir_compare_simple_vs_bittrue, compare_file='y.txt'):
    """
        Compare the simple-golden model with the bit-true model and print summary of the comparison

        Parameters
        ----------
        golden_simple : Integer
            First summand
        golden_bittrue : Integer
            Second summand
        dir_compare_simple_vs_bittrue : Integer
            Upper bound for the addition
        compare_file : Integer
            Lower bound for the addition
    """
    mismatches, total = compare(golden_simple + compare_file,
                                golden_bittrue + compare_file,
                                dir_compare_simple_vs_bittrue + compare_file)

    print("    #################################################")
    print("    ### Correct Results:\t{0}/{1}\n    ### ".format(total - mismatches, total))
    if mismatches == 0:
        print("    ### SUCCESS")
    else:
        print("    ### FAILED:\t{0} ERRORS".format(mismatches))
    print("    #################################################\n")


def add_saturate(a, b, upper_bound, lower_bound):
    """
        Returns the saturated result of an addition of two values a and b

        Parameters
        ----------
        a : Integer
            First summand
        b : Integer
            Second summand
        upper_bound : Integer
            Upper bound for the addition
        lower_bound : Integer
            Lower bound for the addition
    """
    c = int(a) + int(b)
    if c > upper_bound:
        c = upper_bound
    elif c < lower_bound:
        c = lower_bound
    return c


def vadd_saturate(a, b, upper_bound, lower_bound):
    """
        Returns the saturated result of a vector addition of two arrays a and b

        Parameters
        ----------
        a : Array of Integer
            Array of first summands
        b : Array of Integer
            Array of second summands
        upper_bound : Integer
            Upper bound for the addition
        lower_bound : Integer
            Lower bound for the addition
    """
    c = np.zeros(a.shape)
    if isinstance(b, int):
        for i in range(a.shape[0]):
            c[i] = int(a[i]) + int(b)
            if c[i] > upper_bound:
                c[i] = upper_bound
            elif c[i] < lower_bound:
                c[i] = lower_bound
        return c[i]
    if a.shape == b.shape:
        for i in range(a.shape[0]):
            c[i] = int(a[i]) + int(b[i])
            if c[i] > upper_bound:
                c[i] = upper_bound
            elif c[i] < lower_bound:
                c[i] = lower_bound
        return c[i]
    else:
        return np.subtract(c, -1)


def mult_saturate(a, b, upper_bound, lower_bound):
    """
        Returns the saturated result of a multiplication of two values a and b

        Parameters
        ----------
        a : Integer
            Multiplier
        b : Integer
            Multiplicand
        upper_bound : Integer
            Upper bound for the multiplication
        lower_bound : Integer
            Lower bound for the multiplication
    """
    c = float(a) * float(b)
    if c > upper_bound:
        c = upper_bound
    elif c < lower_bound:
        c = lower_bound
    return c


def vmult_saturate(a, b, upper_bound, lower_bound):
    """
        Returns the saturated result of a vector multiplication of two arrays a and b

        Parameters
        ----------
        a : Integer
            Multiplier
        b : Integer
            Multiplicand
        upper_bound : Integer
            Upper bound for the multiplication
        lower_bound : Integer
            Lower bound for the multiplication
    """
    c = np.zeros(a.shape)
    if isinstance(b, int):
        for i in range(a.shape[0]):
            c[i] = int(a[i]) * int(b)
            if c[i] > upper_bound:
                c[i] = upper_bound
            elif c[i] < lower_bound:
                c[i] = lower_bound
        return c[i]
    if a.shape == b.shape:
        for i in range(a.shape[0]):
            c[i] = int(a[i]) * int(b[i])
            if c[i] > upper_bound:
                c[i] = upper_bound
            elif c[i] < lower_bound:
                c[i] = lower_bound
        return c[i]
    else:
        return np.subtract(c, -1)


def get_index(value, bitindex):
    """
        Returns the bit at position `bitindex` of the integer `value`

        Parameters
        ----------
        value : Integer
            Input value
        bitindex : Integer
            Bitindex selector of `value`
    """
    # bitstring = '{0:32b}'.format(value)
    res = int((value >> bitindex) & 0x1)
    if res >= 1:
        return 1
    else:
        return 0


def get_scale(a_index, w_index):
    """
        Returns the proper scale for the BBQ multliplication index

        Parameters
        ----------
        a_index : Integer
            Current activation bit index
        w_index : Integer
            Current weight bit index
    """
    scale = pow(2, (a_index + w_index))
    return scale


def bin2dec(x):
    """
        Returns the binary input into a valid decimal number

        Parameters
        ----------
        x : Integer
            Data in the binary format
    """
    # print(type(x))
    dec = 0
    for i,j in enumerate(x):
        dec += j<<i
    return dec

