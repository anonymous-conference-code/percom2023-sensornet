import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import os
from sklearn.model_selection import train_test_split
from pathlib import Path


def get_frames(df, frame_size, hop_size):
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)
    frames = np.asarray(frames).reshape(-1, 3, frame_size)
    labels = np.asarray(labels)
    return frames, labels


class Prepare_WISDM():
    def prepare_data(self, data_path):
        columns = ['user', 'activity', 'time', 'x', 'y', 'z']
        data = pd.DataFrame(data=None, columns=columns)

        for dirname, _, filenames in os.walk(data_path):
            for filename in filenames:
                df = pd.read_csv(data_path + filename, sep=",", header=None)
                temp = pd.DataFrame(data=df.values, columns=columns)
                data = pd.concat([data, temp])
            data['z'] = data['z'].str.replace(';', '')
            data['x'] = data['x'].astype('float')
            data['y'] = data['y'].astype('float')
            data['z'] = data['z'].astype('float')
        data = data.drop(['user', 'time'], axis=1)
        label = LabelEncoder()
        data['label'] = label.fit_transform(data['activity'])
        x = data[['x', 'y', 'z']]
        y = data['label']
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        scaled_x = pd.DataFrame(data=x, columns=['x', 'y', 'z'])
        # scaled_x['mag'] = np.sqrt(scaled_x['x'] ** 2 + scaled_x['y'] ** 2 + scaled_x['z'] ** 2)
        scaled_x['label'] = y.values
        # scaled_x = scaled_x.drop(['x', 'y', 'z'], axis=1)
        Fs = 20
        frame_size = Fs * 4
        hop_size = Fs * 2
        x, y = get_frames(scaled_x, frame_size, hop_size)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123, stratify=y)
        return [x_train, y_train], [x_test, y_test]
        # return x, y


if __name__ == "__main__":

    # data_path = "./Data/WISDM/phone/accel/"
    root1 = "../Data/WISDM/phone/accel/"

    m = Prepare_WISDM()
    x1, y1 = m.prepare_data(root1)
    print(np.shape(x1[0]))
    print(np.shape(y1[0]))
