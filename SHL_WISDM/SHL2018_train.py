import os
import sys
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

file_path = os.path.realpath(__file__)
project_path = "/".join(file_path.split("/")[:0])
sys.path.insert(0, project_path)


from dataloader_shl.SHL2018_dataloader import create_dataloaders
from architectures.adaptCNN import SensorNet
from architectures.ACmix import SensorNet_ACmix


train_log_dir = os.path.join('runs', 'train')
train_writer = SummaryWriter(log_dir=train_log_dir)
test_log_dir = os.path.join('runs', 'test')
test_writer = SummaryWriter(log_dir=test_log_dir)
"""
SHL2018 dataset training and testing
"""

def seed_torch(seed=123):
    "fix random seeds for reproducibility"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="number of batch size, (default, 128)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning_rate, (default: 1e-3)")
    parser.add_argument('--print-interval', type=int, default=5, help="training information and evaluation information output frequency, (default: 5)")
    parser.add_argument('--model', type=str, default="SensorNet")
    args = parser.parse_args()

    return args

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by xxevery xxx epochs"""
    if epoch == 10000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('accuracy', ':.4e')
    f1_score = AverageMeter('f1_score', ':.4e')

    running_loss = 0.0
    # switch to train mode
    model.train()

    end = time.time()

    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))
        acc = accuracy_(output, target)
        f1 = f1_(output, target) * 100
        accuracy.update(acc, data.size(0))
        f1_score.update(f1, data.size(0))
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        running_loss += loss.item()
        if i % args.print_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                  'f1_score {f1_score.val:.3f} ({f1_score.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, accuracy=accuracy, f1_score=f1_score))

            train_writer.add_scalar("training_loss", running_loss / args.print_interval, epoch * len(train_loader) + i)

            running_loss = 0.0

    return accuracy.avg, f1_score.avg



def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('accuracy', ':.4e')
    f1_score = AverageMeter('f1_score ', ':.4e')
    # progress = ProgressMeter(len(val_loader), losses, accuracy, f1_score,
    #                          prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))
            acc = accuracy_(output, target)
            f1 = f1_(output, target) * 100
            accuracy.update(acc, data.size(0))
            f1_score.update(f1, data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            running_loss += loss.item()
            if i % args.print_interval == 0:
                test_writer.add_scalar("validating_loss", running_loss / args.print_interval,
                                         epoch * len(val_loader) + i)

                running_loss = 0.0

            # TODO: this should also be done with the ProgressMeter
        print(' validate: accuracy {accuracy.avg:.3f} f1_score {f1_score.avg:.3f}'
              .format(accuracy=accuracy, f1_score=f1_score))

        return accuracy.avg, f1_score.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('accuracy', ':.4e')
    f1_score = AverageMeter('f1_score ', ':.4e')
    progress = ProgressMeter(len(test_loader), losses, accuracy, f1_score,
                             prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))
            acc = accuracy_(output, target)
            f1 = f1_(output, target) * 100
            accuracy.update(acc, data.size(0))
            f1_score.update(f1, data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # TODO: this should also be done with the ProgressMeter
        print(' Test: accuracy {accuracy.avg:.3f} f1_score {f1_score.avg:.3f}'
              .format(accuracy=accuracy, f1_score=f1_score))

        return accuracy.avg, f1_score.avg



class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy_(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct * 100 / len(target)


def f1_(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')


def main():
    seed_torch()
    args = get_args()
    # signals_list = ['Acc_x', 'Acc_y', 'Acc_z', 'Acc_norm',
    #                 'Gra_x',  'Gra_y',  'Gra_z',  'Gra_norm',
    #                 'LAcc_x', 'LAcc_y', 'LAcc_z', 'LAcc_norm',
    #                 'Gyr_x',  'Gyr_y',  'Gyr_z',  'Gyr_norm',
    #                 'Mag_x',  'Mag_y',  'Mag_z',  'Mag_norm',
    #                 'Ori_x',  'Ori_y',  'Ori_z',  'Ori_w', 'Ori_norm']

    signals_list = ["Acc_norm", "Gyr_y", "Mag_norm", "Ori_w"]

    train_loader, val_loader, test_loader = create_dataloaders(split="balanced", signals_list=signals_list,
                                                               batch_size=args.batch_size,
                                                               comp_preprocess_first=True,
                                                               use_test=True)

    model = SensorNet(5, 4.9, 100).to(device)

    if args.model == "SensorNet_ACmix":
        model = SensorNet_ACmix(4, 5, 4.9, 100).to(device)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameter')
    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    best_acc_train = 0
    best_f1_train = 0

    best_acc = 0
    best_f1 = 0

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    class_hist = [2302, 2190, 686, 2101, 2475, 2083, 2520, 1953]
    loss_weights = [1 / p for p in class_hist]

    loss_weights = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        acc_train, f1_train = train(train_loader, model, criterion,  optimizer, epoch, args)
        acc, f1 = validate(val_loader, model, criterion, epoch, args)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        is_best_train = acc_train > best_acc_train
        best_acc_train = max(acc_train, best_acc_train)

        if is_best_train:
            best_f1_train = f1_train

        if is_best:
            print('Saving..')
            best_f1 = f1
            state = {
                'model': model.state_dict(),
                'best_acc': best_acc,
                'best_f1': best_f1,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = "best_model"
            torch.save(state, './checkpoint/' + filename + '_ckpt_acmix.t7')

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec, end='')
        print(" Train best accuracy:", best_acc_train, " Train best f1 score:", best_f1_train)
        print(" Validate best accuracy:", best_acc, " Validate  best f1 score:", best_f1)

    print('===> Try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/best_model_ckpt_acmix.t7')
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            print('===> Load last checkpoint data')
        except FileNotFoundError:
            print('Can\'t found autoencoder.t7')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    acc, f1 = test(test_loader, model, criterion)
    print(" Test accuracy:", acc, " Test f1 score:", f1)


if __name__ == "__main__":
    main()
