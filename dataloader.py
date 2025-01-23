import os
import numpy as np


def load_data_de(path, subject):

    dict_load = np.load(os.path.join(path, (str(subject))), allow_pickle=True)
    data = dict_load[()]['sample']
    label = dict_load[()]['label']
    split_index = dict_load[()]["clip"]

    x_tr = data[:split_index]
    x_ts = data[split_index:]
    y_tr = label[:split_index]
    y_ts = label[split_index:]

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }

    return data_and_label

def load_data_de2(path, subject, session):
    pass

def load_data_de3(path, subject):
    pass

def load_data_inde(path, subject):
    pass

def extend_normal(sample):
    for i in range(len(sample)):

        features_min = np.min(sample[i])
        features_max = np.max(sample[i])
        sample[i] = (sample[i] - features_min) / (features_max - features_min)
    return sample



