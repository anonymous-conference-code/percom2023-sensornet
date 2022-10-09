# SENSORNET

The relative root path is where this file lies (the "SLEEP" folder),
please stay in the path for following this instruction.

## Prepare datasets
The original download script comes from [Sleep-EDF-20](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b),

We have made a copy of this file with little edition for convenience.

Download the datasets and prepare them as follows:

Sleep-EDF-20:
```
bash ./prepare_datasets/download_edf20.sh
python ./prepare_datasets/prepare_physionet.py --data_dir ./prepare_datasets --output_dir ./prepare_datasets/edf_20_npz --select_ch "EEG Fpz-Cz"
```

## Training SENSORNET

Sleep-EDF-20:

The `config.json` file is used to update the training parameters.
To perform the standard K-fold crossvalidation, specify the number of folds in `config.json` and run the following:
```
bash ./batch_train.sh 0 ./prepare_datasets/edf_20_npz
```
where the first argument represents the GPU id.

If you want to train only one specific fold (e.g. fold 0), use this command:
```
python SLEEP_train_Kfold_CV.py --device 0 --fold_id 0 --np_data_dir ./prepare_datasets/edf_20_npz
```

## Results
The log file of each fold is found in the fold directory inside the save_dir.  
