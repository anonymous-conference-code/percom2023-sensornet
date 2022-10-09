# SENSORNET
## Prepare datasets
Sussex-Huawei Locomotion 2018

The relative root path is where this file lies (the "SHL_WISDM" folder),
please stay in the path for following this instruction.

Download and unzip the dataset:
```
bash ./dataloader_shl/download_SHL2018.sh
bash ./dataloader_shl/unzip_SHL2018.sh 
```

Copy `./Data/SHL2018/train_order.txt` to `./Data/SHL2018/train`
Copy `./Data/SHL2018/test_order.txt` and `./Data/SHL2018/Label.txt`to `./Data/SHL2018/test`

```
cp './Data/SHL2018/train_order.txt' './Data/SHL2018/train'

cp './Data/SHL2018/test_order.txt' './Data/SHL2018/test'
cp './Data/SHL2018/Label.txt' './Data/SHL2018/test'
```


Save raw data to `.pickle` format to speed up the data loading:
```
python3 ./dataloader_shl/SHL2018_topickle.py
```

WISDM Smartphone and Smartwatch Activity and Biometrics Dataset Data Set

Download and unzip the dataset:
```
bash ./dataloader_wisdm/download_WISDM.sh
bash ./dataloader_wisdm/unzip_WISDM.sh 
```

copy the dataset, and remove the ```.DS_Store``` from the zip file:
```
mkdir './Data/WISDM/phone/'
cp -r './Data/WISDM/wisdm-dataset/raw/phone/accel' './Data/WISDM/phone/'
rm './Data/WISDM/phone/accel/.DS_Store'
```

## Training SENSORNET
```
python3 SHL2018_train.py --model SensorNet
python3 WISDM_train.py --model SensorNet
```
