import torch
import os
import argparse
from torch.autograd import Variable
from model.DA_Net import DA_Net
from UtilCollection.util import compute_F1_score, exponential_decay, save_result, plot_roc, random_seed
from dataloader.read_UEA import load_UEA
import time
#UEA datasets
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
length = 1536 * 2
writer = SummaryWriter('runs/exp')
# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。

parser = argparse.ArgumentParser(description='DA-Net for MTSC')

parser.add_argument('--model', type=str, default='DA-Net')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--length', type=int, default=8192, help='Embedding length')
parser.add_argument('--writer_path', type=str, default='runs/exp', help='TensorBoard path')
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dropout', type=float, default=0.05, help='attention dropout rate')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--cache_path', type=str, default='./cache')
parser.add_argument('--window', type=int, default=64)  # [32,48,64,80,96]
parser.add_argument('--M_name', type=str, default='DA-Net')

args = parser.parse_args()
M_name=args.M_name
writer = SummaryWriter(args.writer_path)  #visualize
random_seed(args.seed)


def GetDataAndNet(archive_path, archive_name, wa, prob, mask=1):
    train_loader, test_loader, num_class = load_UEA(archive, args)

    # get the length and channel of time series
    time_stmp = train_loader.__iter__().next()[0].shape[2]
    in_channel = train_loader.__iter__().next()[0].shape[1]
    # num_class = DealDataset(train_path).num_class()

    net = DA_Net(
        t=time_stmp,
        down_dim=length,
        hidden_dim=(96, 192, 62),
        layers=(2, 2, 6, 2),

        heads=(3, 6, 12,24),
        channels=in_channel,
        num_classes=num_class,
        head_dim=32,
        window_size=args.window,
        downscaling_factors=(4, 2, 2,2),  # 代表多长的时间作为一个特征

        relative_pos_embedding=True,
        wa=wa,
        prob=prob,
        mask=mask,
    ).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = torch.nn.DataParallel(net)
    return train_loader, test_loader, net, num_class


def test(epoch):
    total_pred = torch.tensor([], dtype=torch.int64).to(device)
    total_true = torch.tensor([], dtype=torch.int64).to(device)
    score_list = []
    label_list = []
    total_test_acc = 0
    # for batch_id, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
    for batch_id, (x, y) in enumerate(test_loader):

        x = Variable(x).float().to(device)
        y = Variable(y).to(device)
        net.eval()
        start_time = time.time()
        embedding, encoder, output, pred_y = net(x)
        inference_time = time.time() - start_time

        _, y_pred = torch.max(pred_y, -1)
        total_test_acc += (y_pred.cpu() == y.cpu()).sum().item()

        total_pred = torch.cat([total_pred, y_pred], dim=0)
        total_true = torch.cat([total_true, y], dim=0)

        test_loss = loss_func(pred_y, y.to(torch.long))

        niter = epoch * test_loader.dataset.__len__() + batch_id
        if niter % 10 == 0:
            writer.add_scalar('Test Loss Curve {0}({1})'.format(M_name, length), test_loss.data.item(), niter)

        score_list.extend(pred_y.detach().cpu().numpy())
        label_list.extend(y.cpu().numpy())

    plot_roc( num_class, label_list, score_list, L=length)

    f1_score, precision, recall = compute_F1_score(total_true, total_pred)

    return total_test_acc, f1_score, precision, recall, inference_time, test_loss


def train(optimizer):
    train_time = 0
    max_accuracy = 0
    plot_train_loss = []
    plot_test_loss = []
    plot_train_acc = []
    plot_test_acc = []
    for epoch in range(n_epochs):
        ls = []
        s_time = time.time()
        total_train_acc = 0


        # for batch_id,(x,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for batch_id, (x, y) in enumerate(train_loader):
            #torch ALEXNET
            net.train()
            optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1, 0.90)

            x = Variable(x).float().to(device)
            y = Variable(y).to(device)
            # output 我们需要的 all_sample
            embedding, encoder, output, pred_y = net(x)
            # loss
            loss = loss_func(pred_y, y.to(torch.long))

            _, y_pred = torch.max(pred_y, -1)
            acc_train = (y_pred.cpu() == y.cpu()).sum().item()
            total_train_acc += acc_train
            niter = epoch * train_loader.dataset.__len__() + batch_id

            if niter % 10 == 0:
                writer.add_scalar('Train Loss Curve {0}({1})'.format(M_name, length), loss.data.item(), niter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, y_pred = torch.max(pred_y, -1)
            ls.append(loss)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(total_train_acc / train_loader.dataset.__len__()),
              'time: {:.4f}s'.format(time.time() - s_time))
        plot_train_loss.append(loss.item())
        plot_train_acc.append(total_train_acc / train_loader.dataset.__len__())
        train_time += time.time() - s_time

        # print("Total time elapsed: {:.4f}s".format(train_time))
        total_test_acc, f1_score, precision, recall, inference_time, test_loss = test(epoch)
        plot_test_loss.append(test_loss.cpu().detach())
        plot_test_acc.append(total_test_acc / test_loader.dataset.__len__())

        # save model
        if os.path.exists(f'saved_model/{M_name}') == False:
            os.makedirs(f'saved_model/{M_name}')

        if total_test_acc > max_accuracy:
            print('save best model')
            max_accuracy = total_test_acc
            torch.save(net,
                       f'saved_model/{M_name}/{archive} batch={args.batch_size} length={length} window={args.window}.pkl')

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_test: {:.8f}'.format(test_loss.item()),
              'acc_test: {:.4f}'.format(total_test_acc / test_loader.dataset.__len__()),
              'time: {:.4f}s'.format(time.time() - s_time))
    plt.plot()

    if os.path.exists(f'result') == False:
        os.makedirs(f'result')
    save_result(file, ls[-1], total_test_acc / test_loader.dataset.__len__(), f1_score, precision, recall, train_time,
                inference_time, args.window, length)



wa=1
prob=1
if __name__ == '__main__':

    # archives = glob.glob(r'D:/FTP/chengrj/time_series/data/Multivariate_arff/*')
    # for archive_path in archives:
    # archive = os.path.split(archive_path)[-1]

    archive = 'FaceDetection'
    # archive = 'PEMS-SF'
    print(archive)
    file = r'./result/result_{0}.csv'.format(archive)
    train_loader, test_loader, net, num_class = GetDataAndNet(0, archive, wa, prob)

    # for param in net.parameters():
    #     print(param)
    # print(np.sum([np.prod(x.size()) for x in net.parameters()]))

    LEARNING_RATE = 0.001
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=10,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    COMPUTE_TRN_METRICS = True
    n_epochs = args.n_epochs

    loss_func = torch.nn.CrossEntropyLoss()

    train(optimizer)
    # except:
    #     file = r'./result/result_{0}_{1}_{2}_{3}.csv'.format(str(wa), str(prob), str(mask),
    #                                                          archive)
    #     with open(file, 'a+') as f:
    #         f.write('error\n')
    #     continue




# plt.plot(range(len(plot_train_loss)),plot_train_loss,label='train_loss')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# plt.plot(range(len(plot_train_acc)),plot_train_acc,label='train_acc')
# plt.plot(range(len(plot_test_acc)),plot_test_acc,label='test_acc')
# plt.xlabel('iteration')
# plt.ylabel('acc')
# plt.legend()
# plt.show()

