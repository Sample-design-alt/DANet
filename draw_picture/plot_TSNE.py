import sys
sys.path.append('..')
import torch
from dataloader import read_UEA
from UtilCollection.util import random_seed
from torch.autograd import Variable
from UtilCollection.tsne import gather_by_tsne
from UtilCollection.tsne import gather_all_by_tsne
import numpy as np
from dataloader.read_UEA import load_UEA
import argparse
# from draw_picture import grad_cam


print('当前使用的pytorch版本：', torch.__version__)

parser = argparse.ArgumentParser(description='WiWo transformer for TSC')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--length', type=int,default=8192, help='Embedding length')
parser.add_argument('--writer_path', type=str, default='runs/exp', help='TensorBoard path')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--seed', type=int, default=40, help='random seed')
parser.add_argument('--dropout', type=float, default=0.05, help='attention dropout rate')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--cache_path', type=str, default='../cache')
parser.add_argument('--model_path', type=str, default='../saved_model/WiWo transformer/PEMS-SF batch=4 length=3072 window=64.pkl')

args = parser.parse_args()

random_seed(args.seed)

file_name = args.model_path.split('/')[-1].split(' ')[0]

# path = f'./data/MTS_dataset/ECG/ECG.mat'

heatMap_or_not = False  # 是否绘制Score矩阵的HeatMap图
gather_or_not = True  # 是否绘制单个样本的step和channel上的聚类图
gather_all_or_not = True  # 是否绘制所有样本在特征提取后的聚类图

net = torch.load(args.model_path, map_location=torch.device(args.device))

correct = 0
total = 0
train_loader, test_loader, num_class = load_UEA(file_name, args)
with torch.no_grad():
    all_sample_X = []
    all_sample_Y = []
    for x, y in test_loader:
        x = Variable(x).float().to(args.device)
        y = Variable(y).to(args.device)
        embedding, encoder, output, pred_y = net(x)

        all_sample_X.append(output)
        all_sample_Y.append(y)

        if gather_or_not:
            # for index, sample in enumerate(test_loader.max_length_sample_inTest):
            #     if sample.numpy().tolist() in x.numpy().tolist():
            #         target_index = x[0].cpu().detach().numpy().tolist()
                    print('正在绘制gather图...')
                    gather_by_tsne(embedding[0].cpu().detach().numpy(), np.arange(embedding[0].shape[1]),
                                   3, file_name + ' input_gather')
                    print('gather图绘制完成！')
                    # draw_data = x[target_index].transpose(-1, -2)[0].cpu().detach().numpy()
                    # draw_colorful_line(draw_data)
                    gather_or_not = False

        _, label_index = torch.max(pred_y, dim=-1)
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()

    if gather_all_or_not:
        all_sample_X = torch.cat(all_sample_X, dim=0).cpu().detach().numpy()
        all_sample_Y = torch.cat(all_sample_Y, dim=0).cpu().detach().numpy()
        print('正在绘制gather图...')
        gather_all_by_tsne(all_sample_X, all_sample_Y, num_class, file_name + ' all_sample_gather')
        print('gather图绘制完成！')

    print(f'Accuracy: %.2f %%' % (100 * correct / total))
