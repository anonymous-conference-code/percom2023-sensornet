import os
import sys
import torch
import numpy as np

file_path = os.path.realpath(__file__)
project_path = "/".join(file_path.split("/")[:-1])
sys.path.insert(0, project_path)


from SHL2018_param import device
import SHL2018_dataset



class ConcatCollate():
    def __init__(self, list_signals=None, flag_debug=False):

        self.flag_debug = flag_debug
        self.list_signals = list_signals

    def __call__(self, L):

        if self.list_signals == None:
            # take the signals of the first sample
            signal_name_list = list(L[0][0].keys())
        else:
            signal_name_list = self.list_signals

        batch_size = len(L)

        x_example = L[0][0] # a dict relative to first instance
        y_example = L[0][1]
#        signal_name_list = list(x_example.keys())  # list of sensor name
        n_signals = len(signal_name_list)
        signal_name_example = signal_name_list[0] # string

        instance = x_example[signal_name_example] # array (1,n) or (1,nb_f, nb_t)

        flag_debug = self.flag_debug
        axis_concat = 0

        signal_shape = instance.shape
        batch_shape = (batch_size,) + signal_shape
        batch_shape = list(batch_shape)  # tuples cannot be modified in-place
        batch_shape[axis_concat+1] *= n_signals   # for the final tensor, the first axis is devoted to the batch, hence the +1
        batch_shape = tuple(batch_shape)

        if flag_debug:
            print('X_batch shape')
            print(batch_shape)

        X_batch = torch.zeros(batch_shape, dtype=torch.float32, device=device)
        Y_batch = torch.zeros(batch_size,  dtype=y_example.dtype, device=device)

        for index_batch, (instance, y) in enumerate(L):

            signal_list = [instance[signal_name] for signal_name in signal_name_list] # all signals for the given instance

            torch.cat( signal_list, dim=axis_concat, out=X_batch[index_batch,...] )

            Y_batch[index_batch] = y

        # sending the data to cuda
        Y_batch = Y_batch-1
        return X_batch, Y_batch


class TemporalTransform():
    """  create the base transform to use to each element of the data
    Also generates data for thr 'Random' and 'Zero' cases
    Parameters
    ----------
    signal_name_list: a list of string signals (ex: 'Gyr_y', 'Ori_x')
        If a string ends by "_norm" (ex: "Mag_norm"), the output will
        be the norm of the three (or four) axis of the signal.
        The signals can also be 'Zero', in which case the segments are only
        zeros, or 'Random', in which case each data oint is sampled with normal
        distribution (zero mean, unit variance)
    Returns
    -------
    a function with input:  a dict of (_, 6000) arrays (key example: 'Gyr_y')
                and output: a dictionnary of arrays.
    """
    def __init__(self, signal_name_list):
        super(TemporalTransform, self).__init__()
        self.signal_name_list = signal_name_list

    def __call__(self, data):
        """
        Parameters
        ----------
        a dict of (B, 6000) arrays (key example: 'Gyr_y')
        Returns
        -------
        a dictionnary of arrays. This time, the keys are from signal_name_list,
            and the values are either raw signals (if the key ends with '_x',
            '_y', '_z', or '_w'); a norm of several signals (if the key ends
            with '_norm'); or a specific signal (if the key is 'Random' or
            'Zero'). The shape of each array is (B, 6000), where B (batch size)
            depends on the input shape.
        """

        outputs = {}
        for signal_name in self.signal_name_list:

            if signal_name[-2:] in ['_x', '_y', '_z', '_w'] or signal_name == "Pressure":
                processed_signal = data[signal_name]

            elif signal_name == 'Random':
                data_shape = data["Acc_x"].shape
                processed_signal = np.random.randn(data_shape[0], data_shape[1]).astype(np.float32)

            elif signal_name == 'Zero':
                data_shape = data["Acc_x"].shape
                processed_signal = np.zeros(data_shape).astype(np.float32)

            elif signal_name[-5:] == '_norm':
                suffix_location = signal_name.index("_") # 4 if signal_name == "LAcc", 3 otherwise
                sensor = signal_name[:suffix_location]    # ex: 'Acc', 'LAcc'

                if sensor == "Ori":
                    # in that case, data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2 should be 1.0
                    processed_signal = np.sqrt(data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2 \
                                             + data[sensor+"_w"]**2)
                else :
                    processed_signal = np.sqrt(data[sensor+"_x"]**2 + data[sensor+"_y"]**2 + data[sensor+"_z"]**2)
            else :
                raise ValueError("unknown signal name: '{}'. Signal names should end with either '_x', '_y', '_z', '_w', or '_norm'".format(signal_name))

            outputs[signal_name] = processed_signal

        return outputs

    def __str__(self):
        """purely for visual purposes, so that we can print() the function"""
        str_to_return = "Temporal_transform"
        str_to_return += "\n\t Signals: {}".format(self.signal_name_list)
        return str_to_return


def duplicate_in(l):
    """Returns True if there is at least one duplicated element in the list of
    strings l, and False otherwise"""
    l.sort()
    for i in range(len(l)-1):
        if l[i] == l[i+1]: return True
    return False


if __name__ == '__main__':
    assert (duplicate_in(['Acc_norm', 'Acc_x', 'Mag_z']) == False)
    assert (duplicate_in(['Acc_norm', 'Acc_x', 'Acc_norm']) == True)


def remove_duplicates(l):
    "Create a new list that contains the same elements as l, without duplicates"
    result = []
    for element in l:
        if element not in result:
            result.append(element)
    return result


if __name__ == '__main__':
    assert (remove_duplicates(['Acc_norm', 'Acc_x', 'Mag_z'])    == ['Acc_norm', 'Acc_x', 'Mag_z'] )
    assert (remove_duplicates(['Acc_norm', 'Acc_x', 'Acc_norm']) == ['Acc_norm', 'Acc_x'])


def create_dataloaders(split, signals_list, batch_size, comp_preprocess_first=True, use_test=False):

    print("create_dataloaders", signals_list)

    transform_fn = TemporalTransform(remove_duplicates(signals_list))

    collate_fn = ConcatCollate(list_signals=signals_list)



    train_dataset = SHL2018_dataset.SignalsDataSet(mode='train', split=split, comp_preprocess_first=comp_preprocess_first, transform=transform_fn)
    val_dataset =  SHL2018_dataset.SignalsDataSet(mode='val',   split=split, comp_preprocess_first=comp_preprocess_first, transform=transform_fn)

    if use_test:
        test_dataset =SHL2018_dataset.SignalsDataSet(mode='test',  split=split, comp_preprocess_first=comp_preprocess_first, transform=transform_fn)


     # we need full-rank correlation matrices estimation for deep CCA
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, collate_fn=collate_fn, shuffle=use_test)
    if use_test:
        test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    else:
        test_dataloader = []

    return train_dataloader, val_dataloader, test_dataloader
