import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import data
import models
import utils

parser = argparse.ArgumentParser(description='FGE training')

parser.add_argument('--dir', type=str, default='/tmp/fge/', metavar='DIR',
                    help='training directory (default: /tmp/fge)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--cycle', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--lr_1', type=float, default=0.05, metavar='LR1',
                    help='initial learning rate (default: 0.05)')
parser.add_argument('--lr_2', type=float, default=0.0001, metavar='LR2',
                    help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

assert args.cycle % 2 == 0, 'Cycle length should be even'

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'fge.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

checkpoint = torch.load(args.ckpt)
start_epoch = checkpoint['epoch'] + 1
model.load_state_dict(checkpoint['model_state'])
model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_1,
    momentum=args.momentum,
    weight_decay=args.wd
)
optimizer.load_state_dict(checkpoint['optimizer_state'])

ensemble_size = 0
predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'ens_acc', 'time']

for epoch in range(args.epochs):
    time_ep = time.time()
    lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
    train_res = utils.train(loaders['train'], model, optimizer, criterion, lr_schedule=lr_schedule)
    test_res = utils.test(loaders['test'], model, criterion)
    time_ep = time.time() - time_ep
    predictions, targets = utils.predictions(loaders['test'], model)
    ens_acc = None
    if (epoch % args.cycle + 1) == args.cycle // 2:
        ensemble_size += 1
        predictions_sum += predictions
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

    if (epoch + 1) % (args.cycle // 2) == 0:
        utils.save_checkpoint(
            args.dir,
            start_epoch + epoch,
            name='fge',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy'], ens_acc, time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
