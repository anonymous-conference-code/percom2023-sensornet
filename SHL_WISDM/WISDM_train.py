import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader_wisdm.WISDM_dataloader import data_generator_np
from dataloader_wisdm.WISDM_dataset import Prepare_WISDM
from sklearn.metrics import f1_score

from architectures.adaptCNN import SensorNet
from architectures.ACmix import SensorNet_ACmix


if __name__ == '__main__': # this is used to launch the file from anywhere
    import sys
    sys.path.append("..")

train_log_dir = os.path.join('runs', 'train')
train_writer = SummaryWriter(log_dir=train_log_dir)
test_log_dir = os.path.join('runs', 'test')
test_writer = SummaryWriter(log_dir=test_log_dir)

"""
WISDM dataset training and testing
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


def adjust_learning_rate(optimizer, epoch, args):
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
    running_acc = 0.0
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
        running_acc += acc

        if i % args.print_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                  'f1_score {f1_score.val:.3f} ({f1_score.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, accuracy=accuracy, f1_score=f1_score))

            train_writer.add_scalar("training_loss", running_loss / args.print_interval, epoch * len(train_loader) + i)
            train_writer.add_scalar("training_accuracy", running_acc / args.print_interval, epoch * len(train_loader) + i)

            running_loss = 0.0
            running_acc = 0.0

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
    running_acc = 0.0

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
            running_acc += acc

            if i % args.print_interval == 0:
                test_writer.add_scalar("validating_loss", running_loss / args.print_interval,
                                         epoch * len(val_loader) + i)
                test_writer.add_scalar("validating_accuracy", running_acc / args.print_interval,
                                       epoch * len(val_loader) + i)


                running_loss = 0.0
                running_acc = 0.0

            # TODO: this should also be done with the ProgressMeter
        print(' validate: accuracy {accuracy.avg:.3f} f1_score {f1_score.avg:.3f}'
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
    root1 = "./Data/WISDM/phone/accel/"
    args = get_args()
    m = Prepare_WISDM()
    training_files, subject_files = m.prepare_data(root1)

    train_loader, test_loader = data_generator_np(training_files, subject_files, args.batch_size)


    # model = SensorNet(2, 1.5, 20).to(device)

    if args.model == "SensorNet_ACmix":
        model = SensorNet_ACmix(3, 2, 1.5, 20).to(device)

    elif args.model == "SensorNet":
        model = SensorNet(2, 1.5, 20).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameter')
    if torch.cuda.device_count() >= 1:
        print("num GPUs: ", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)

    best_acc = 0
    best_f1 = 0

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    start_time = time.time()
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion,  optimizer, epoch, args)
        acc, f1 = validate(test_loader, model, criterion, epoch, args)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if is_best:
            print('Saving..')
            best_f1 = f1
            state = {
                'model': model.state_dict(),
                'best_acc1': best_acc,
                'best_acc5': best_f1,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = "best_model"
            torch.save(state, './checkpoint/' + filename + '_ckpt_wisdm.t7')

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec, end='')
        print(" Test best accuracy:", best_acc, " Test best f1 score:", best_f1)


if __name__ == "__main__":
    main()
