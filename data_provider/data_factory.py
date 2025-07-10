from data_provider.data_loader import PSMSegLoader, MSLSegLoader, SMDSegLoader, SWATSegLoader,UCRSegLoader,SMAPSegLoader
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from data_provider.ucr import UCR
import merlion
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize
import ast

'''
dataset_name SMD
(708405, 38) (708420, 38)
dataset_name MSL
(58317, 55) (73729, 55)
dataset_name PSM
(132481, 25) (87841, 25)
dataset_name SWAT
(495000, 51) (449919, 51)
'''


def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    # salesforce-merlion==1.1.1
    bias, scale = mvn.bias, mvn.scale

    train_time_series = train_time_series_ts.to_pd().to_numpy()

    # train_data = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()

    train_data = train_time_series
    test_data = test_time_series
    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


def Distribute_data(dataset_name, root_path, flag,n):  
    r = 1
    if dataset_name == 'SMD':
        train_data = np.load(os.path.join(root_path, "SMD_train.npy"))
        train_labels = train_data
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))
        if r<1:
            train_data = train_data[:int(r * len(train_data))]
            train_labels = train_labels[:int(r * len(train_labels))]
            test_data = test_data[:int(r * len(test_data))]
            test_labels = test_labels[:int(r * len(test_labels))]

      
    elif dataset_name == 'MSL':
        train_data = np.load(os.path.join(root_path, "MSL_train.npy"))
        train_labels = train_data
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        if r<1:
            train_data = train_data[:int(r * len(train_data))]
            train_labels = train_labels[:int(r * len(train_labels))]
            test_data = test_data[:int(r * len(test_data))]
            test_labels = test_labels[:int(r * len(test_labels))]
        
    elif dataset_name == 'PSM':
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        train_data = np.nan_to_num(data)
        train_labels = train_data
        test_data =  pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        if r<1:
            train_data = train_data[:int(r * len(train_data))]
            train_labels = train_labels[:int(r * len(train_labels))]
            test_data = test_data[:int(r * len(test_data))]
            test_labels = test_labels[:int(r * len(test_labels))]
        
    elif dataset_name == 'SWAT':
        data = pd.read_csv(os.path.join(root_path, 'swat_train.csv'))
        train_data = data.values[:, :-1]
        train_labels = train_data
        test_data = pd.read_csv(os.path.join(root_path, 'swat_test.csv'))
        test_labels = test_data.values[:, -1:]
        test_data = test_data.values[:, :-1]
        if r<1:
            train_data = train_data[:int(r * len(train_data))]
            train_labels = train_labels[:int(r * len(train_labels))]
            test_data = test_data[:int(r * len(test_data))]
            test_labels = test_labels[:int(r * len(test_labels))]
    
    elif dataset_name == 'SMAP':
        train_data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        train_labels = train_data
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))

    elif dataset_name == 'UCR':
        dt = UCR()
        train_data_list = []
        test_data_list = []
        train_labels_list = []
        test_labels_list = []
        for i in range(n):
            time_series, meta_data = dt[i]
            train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)
       
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            train_labels_list.append(train_labels)
            test_labels_list.append(test_labels)

        train_data = np.concatenate(train_data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
    print("test:", test_data.shape)
    print("train:", train_data.shape)
    
    return train_data, train_labels, test_data, test_labels


def data_provider(args, flag, dataset_type):
    data_dict = {
    'SMD': SMDSegLoader,
    'MSL': MSLSegLoader,
    'PSM': PSMSegLoader,
    'SWAT': SWATSegLoader,
    'SMAP': SMAPSegLoader,
    'UCR': UCRSegLoader
    }
    Data = data_dict[dataset_type]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True 
        if args.task_name == 'anomaly_detection':
            batch_size = args.batch_size
        else:
            batch_size = 1 
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq 

    drop_last = False              
 
    root_path = os.path.join(args.root_path, dataset_type) 
    train_data, train_labels, test_data, test_labels = Distribute_data(dataset_type, root_path, flag, args.n_ucr if hasattr(args, 'n_ucr') else 1)
 
    base_params = {
        'data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'root_path': root_path,
        'win_size': args.seq_len,
        'flag': flag
    }
    additional_params = {
            'patch_len': 10,
            'patch_stride': 10
    }
    if dataset_type == 'SMD':
        additional_params['step'] = 100
    else:
        additional_params['step'] = 1
   
    params = {**base_params, **additional_params}
    data_set = Data(**params)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        persistent_workers=False
    ) 
    
    return data_set, data_loader
