#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script save raw data to .pickle format in [data_path]/{train|test}/ordered_data.pickle
Expected file tree :
[data_path]
    train
        Acc_x.txt
        Acc_y.txt
        Acc_z.txt
        Gyr_x.txt
        etc.
    test
        Acc_x.txt
        etc.
The saved data is a dictionary with keys such as "Acc_x", and numpy arrays as * values (shapes: (N_samples, 6000)).
We use .pickle instead of .txt files because loading the dataset is ~40 times faster.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import pickle
from pathlib import Path
from SHL2018_param import classes_names, data_path
import time

"""
set the debug here!
"""
debug = 0
# if debug == 1, apply the same treatment to train_order.txt instead
# of real data (Acc_x, Gyr_y, etc.). It will allow us to check everything
# is here, and that the segments split according to their order.
# be careful, debug == 1 *overwrites* the data on the disk

if debug == 0:
    base_signals = [signal + "_" + coord for signal in ["Acc", "Gra", "LAcc", "Gyr", "Mag", "Ori"]
                    for coord in ["x", "y", "z"]] + ["Ori_w", "Pressure"]
    # you may change here depending on what signals you want to keep
elif debug == 1:
    base_signals = ["train_order"]
    print("=============================== Debug mode ===============================")

segment_size = 6000 if debug == 0 else 1
# the number of points in a sample. In real SHL 2018 samples, there are 6000 points,
# but train_order only yields one integer per line

start_time = time.time()


if __name__ == "__main__":
    for mode in ["train", "test"]:
        """
        ['Acc_x', 'Acc_y', 'Acc_z', 'Gra_x', 'Gra_y', 'Gra_z', 
        'LAcc_x', 'LAcc_y', 'LAcc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z', 
        'Mag_x', 'Mag_y', 'Mag_z', 'Ori_x', 'Ori_y', 'Ori_z', 'Ori_w', 
        'Pressure', 'Label']
        """
        chosen_signals = base_signals + ["Label"] if debug == 0 else base_signals

        print("\n ---- {} mode ----".format(mode))
        print('chosen signals: ', chosen_signals)

        """ 
        ./Data/SHL2018/train
        ./Data/SHL2018/test 
        """
        mode_path = data_path / Path(mode)

        """Principle: we load all the data into a dictionary of numpy arrays,
        before writing the values in a pickle. We look at each of the samples,
        and append the sample to the line corresponding to its order"""

        if mode == "train":
            order_array = np.loadtxt(mode_path / Path("train_order.txt")).astype(int)
        elif mode == "test":
            order_array = np.loadtxt(mode_path / Path("test_order.txt")).astype(int)

        ref_file = mode_path / Path(chosen_signals[0] + ".txt")

        with open(ref_file) as f:
            len_file = sum(1 for row in f)  # train: 16310 / test: 5698

        """
        create two dictionaries
    
        file_dict:
        {'Acc_x': PosixPath('./Data/SHL2018/train/Acc_x.txt'), ...}
    
        data_dict:
        {'Acc_x': array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), ...}
        """
        file_dict = {signal: mode_path / Path(signal + ".txt") for signal in chosen_signals}

        data_dict = {signal: np.zeros((len_file, segment_size), dtype='float32') for signal in chosen_signals}

        for signal in chosen_signals:
            print("\n", signal)
            print("loading '%s'... " % file_dict[signal], end='')
            original_data = np.loadtxt(file_dict[signal])  # load the raw data from .text
            if debug:  # in debug mode, only the order is loaded, which has only one dimension
                original_data = original_data[:, None]
            print('Done')

            len_this_file, segment_size = original_data.shape
            print("%d samples, each sample having %d points" % (len_file, segment_size))  # 16310 x6000 / 5698 x 6000
            # sanity check : we stop if the number of lines in one file differs from the rest
            assert (len_this_file == len_file), "Two files contain a different number of segments"

            for line_index in range(len_file):
                order = order_array[line_index] - 1
                data_dict[signal][order, :] = original_data[line_index, :]  # arrange data in order

        data_dict['order'] = order_array

        if debug == 0:
            Labels = data_dict["Label"]
            n_changes = (Labels[:-1, -1] != Labels[1:, 0]).sum()
            print("""Reordering verification: \nthere are {} modes changes between one segment and the following
                         (should be small compared to the {} segments in total)""".format(n_changes, len_file))

            # saving the values
        filepath = mode_path / Path("ordered_data.pickle")
        with open(filepath, "wb") as f:
            pickle.dump(data_dict, f)

        # %%    Verification using train_order.txt
        if debug and mode == "train":
            del data_dict

            # test 1 : check everything is at the good location
            filepath = mode_path / Path("ordered_data.pickle")
            with open(filepath, "rb") as f:
                data_dict = pickle.load(f)

            print("the data has been saved and reloaded")

            # test 2 : check everything is here
            order_values = data_dict["train_order"]
            order_values = [int(o) for o in order_values]
            sorted_order_values = sorted(order_values)
            expected_list = [int(o) + 1 for o in
                             range(len_file)]  # keep in mind the elements of order_values are between 1 and n, included
            if sorted_order_values == expected_list:
                print("all values are here")
            else:
                # either some values are missing, or we got some unexpected values (or both)
                missing_values = [o for o in expected_list if o not in order_values]
                print("missing order values: ", missing_values)
                unexpected_values = [o for o in order_values if o not in expected_list]
                print("some order values are not expected: ", unexpected_values)

            # test 3: check the data is sorted
            if order_values == sorted(order_values):
                print("the reordering is correct")
            else:
                print("there has peen a problem in the reordering. The order values should be sorted, they are:")
                print([int(o) for o in order_values])

            break  # do not go to test mode

        # end for mode
    print('\n processing time : %.3f min' % ((time.time() - start_time) / 60))



