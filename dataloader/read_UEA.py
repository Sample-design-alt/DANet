import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


def load_UEA(archive_name, args):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    # load from cache
    cache_path = f'{args.cache_path}/{archive_name}.dat'
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)


    # load from arff
    else:
        train_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)


    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)
    # return TrainDataset,TestDataset,len(labels)
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_loader, test_loader, num_class


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))


if __name__ == '__main__':
    load_UEA('Ering')
