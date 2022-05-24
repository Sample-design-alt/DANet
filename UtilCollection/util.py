import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from itertools import cycle
from numpy import interp
import random, os
import seaborn as sns
# sns.set()

def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    if (staircase):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
    else:
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

    return optimizer


def compute_F1_score(trueY, predY):
    oriF1 = f1_score(trueY.cpu().data.numpy(), predY.cpu().data.numpy(), average='macro')
    precision = precision_score(trueY.cpu().data.numpy(), predY.cpu().data.numpy(), average='macro')
    recall = recall_score(trueY.cpu().data.numpy(), predY.cpu().data.numpy(), average='macro')

    return oriF1, precision, recall


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, filename):
        data = np.loadtxt(filename, delimiter='\t')
        Y = data[:, 0]
        X = data[:, 1:]

        Y = preprocessing.LabelEncoder().fit(Y).transform(Y)
        X[np.isnan(X)] = 0
        X = preprocessing.scale(X)
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(Y)
        self.len = X.shape[0]
        if len(self.x_data.shape) == 2:  # 单元时间序列要增加一个维度
            self.x_data = torch.unsqueeze(self.x_data, 1)
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))


def save_result(file, loss, accuracy, f1_score, precision, recall, train_time, inference_time, window, length):
    print('accuracy:', accuracy)
    with open(file, 'a+') as f:
        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(str(loss.item()), str(accuracy), f1_score, precision,
                                                               recall, train_time, inference_time, window, length))
    print('parameters have been saved to {0}'.format(file))


def random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def plot_roc(name, num_class, label_list, score_list, L):
    score_array = np.array(score_list)
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_tensor = label_tensor.to(torch.int64)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()

    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'khaki', 'lavender', 'lemonchiffon', 'lightgreen', 'lightsteelblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    # plt.title('Some extension of ROC to {0}'.format(name))
    # plt.title('Some extension of ROC to DA-Net', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="lower right", fontsize=15)
    plt.savefig('Image/{0}_window_64.pdf'.format(name), format='pdf')
    # plt.show()
