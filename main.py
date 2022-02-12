import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import re
from scipy.io import loadmat
from scipy.io import savemat
from os import walk

from DSAN import DSAN
import data_loader


def load_data(root_path, src, tar, batch_size, test_subj_index):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    # 输入 sample
    filenames = next(walk('D:/wzw/DSAN/seizure5times/data/'), (None, None, []))[2]  # [] if no file

    index_arr = [1, 5, 7, 13, 14, 15, 16, 17, 19, 20, 22, 23, 25, 33, 38, 39, 40, 41, 44, 50, 54, 63, 64, 67, 69, 71, 73, 78, 79]
    root_dir = 'D:/wzw/DSAN/seizure5times/data/'
    # 输入 label
    f_label = loadmat('D:/wzw/DSAN/seizure5times/label/addlabel.mat')
    arr_label = f_label['y']


    f = []
    for i in range(len(filenames)):

        # 解决 sample和 label index不符的问题
        string = filenames[i]
        pattern = re.compile(r'(?<=eeg)\d+\.?\d*')
        # 当前处理的 patient ID
        ind = int(pattern.findall(string)[0])

        # 找到对应的 label 索引
        actual_ind = index_arr.index(ind)
        #print(actual_ind)
        labels = arr_label[:,actual_ind] # (15416,)

        curr_file_path = root_dir + filenames[i]
        m = loadmat(curr_file_path)
        arr_sample = m['x']
        labels = labels[:arr_sample.shape[0]] # remove unused labels

        # 用 sample 和 label create tuple
        #print(arr_sample.shape, labels.shape)
        new_tuple = (arr_sample, labels)
        # 输出的 f 中包含29个 tuple
        f.append(new_tuple)

    # 构造source 和 target domain  TODO source domain is from per patient but not all 28 patients.
    src = f.copy()
    # 每次一个 patient 作为目标域 在目标域划分训练集和测试集
    tar = src.pop(test_subj_index)
    # TODO split valid set
    index_train = int(0.6*tar[0].shape[0])
    # index_test = tar.shape[0] -  int(0.6*tar.shape[0])
    tar_train = (tar[0][0:index_train, :, :], tar[1][0:index_train])
    tar_test = (tar[0][index_train:, :, :], tar[1][index_train:])

    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_testing(root_path, tar_train, batch_size, kwargs, True)
    loader_tar_test = data_loader.load_testing(root_path, tar_test, batch_size, kwargs, False)

    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    

    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter): # TODO num_iter
        tmp_dict = iter_source.next()
        data_source, label_source = tmp_dict['feature'], tmp_dict['label']
        tmp_dict = iter_target.next()
        data_target = tmp_dict['feature']
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source, label_source
        data_target = data_target

        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)


        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source.long())
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_lmmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        iter_target = iter(dataloader)
        dict = next(iter_target)
        zero_cnt_true, zero_cnt_false, ones_cnt_true, ones_cnt_false, twos_cnt_true, twos_cnt_false = 0, 0, 0, 0, 0, 0
        while dict != None:
            data, target = dict['feature'], dict['label']
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target.long()).item()
            pred = pred.data.max(1)[1]

            bool_tensor = pred.eq(target.data.view_as(pred))

            for i in range(len(bool_tensor)):
                if bool_tensor[i] == True and target.data.view_as(pred)[i] == 0:
                    zero_cnt_true += 1
                elif bool_tensor[i] == False and target.data.view_as(pred)[i] == 0:
                    zero_cnt_false += 1
                if bool_tensor[i] == True and target.data.view_as(pred)[i] == 1:
                    ones_cnt_true += 1
                elif bool_tensor[i] == False and target.data.view_as(pred)[i] == 1:
                    ones_cnt_false += 1
                if bool_tensor[i] == True and target.data.view_as(pred)[i] == 2:
                    twos_cnt_true += 1
                elif bool_tensor[i] == False and target.data.view_as(pred)[i] == 2:
                    twos_cnt_false += 1

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            dict = next(iter_target, None)

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct, zero_cnt_true, zero_cnt_false, ones_cnt_true, ones_cnt_false, twos_cnt_true, twos_cnt_false


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='-1')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='-1')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='-1')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=3)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=64)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.0001, 0.001, 0.001])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    #torch.set_default_dtype(torch.float64)
    args = get_args()
    print(vars(args))
    #device = torch.device('cpu')
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda')
    patient_num = 29
    acc_arr = []
    class_acc_arr = []
    for i in range(patient_num): # TODO
        print('patient_num:', i)
        dataloaders = load_data(args.root_path, args.src,
                                args.tar, int(args.batch_size), i)
        model = DSAN(num_classes=args.nclass)

        correct = 0
        stop = 0

        if args.bottleneck:
            optimizer = torch.optim.SGD([
                {'params': model.feature_layers.parameters()},
                {'params': model.bottle.parameters(), 'lr': args.lr[1]},
                {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
            ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
        else:
            optimizer = torch.optim.SGD([
                {'params': model.feature_layers.parameters()},
                {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
            ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

        for epoch in range(1, args.nepoch + 1):
            print('epoch:', epoch)
            stop += 1
            for index, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

            train_epoch(epoch, model, dataloaders, optimizer)
            t_correct, zero_cnt_true, zero_cnt_false, ones_cnt_true, ones_cnt_false, twos_cnt_true, twos_cnt_false = test(model, dataloaders[-1])
            if t_correct > correct:
                correct = t_correct
                stop = 0
                torch.save(model, 'model.pkl')
            print(
                f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

            if stop >= args.early_stop or epoch == args.nepoch:
                print(
                    f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
                acc = 100. * correct / len(dataloaders[-1].dataset)
                zero_cnt_true, zero_cnt_false, ones_cnt_true, ones_cnt_false, twos_cnt_true, twos_cnt_false
                zero_accuracy = 100. * zero_cnt_true / (zero_cnt_true + zero_cnt_false)
                ones_accuracy = 100. * ones_cnt_true / (ones_cnt_true + ones_cnt_false)
                twos_accuracy = 100. * twos_cnt_true / (twos_cnt_true + twos_cnt_false)
                acc_arr.append(acc)
                class_acc_arr.append([zero_accuracy, ones_accuracy, twos_accuracy])
                print(acc)
                print([zero_accuracy, ones_accuracy, twos_accuracy])
                break


    print(acc_arr)
    print(class_acc_arr)



