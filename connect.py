import argparse
import numpy as np
import os
import sys
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import utils

parser = argparse.ArgumentParser(description='Connect models with polychain')

parser.add_argument('--dir', type=str, default='/tmp/chain/', metavar='DIR',
                    help='training directory (default: /tmp/chain)')

parser.add_argument('--num_points', type=int, default=6, metavar='N',
                    help='number of points between models (default: 6)')

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
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT', required=True,
                    help='checkpoint to eval, pass all the models through this parameter')


parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'chain.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False
)

architecture = getattr(models, args.model)
base_model = architecture.base(num_classes, **architecture.kwargs)
base_model.cuda()


criterion = F.cross_entropy
regularizer = utils.l2_regularizer(args.wd)


def get_weights(model):
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])

T = (args.num_points - 1) * len(args.ckpt - 1) + 1
ts = np.linspace(0.0, len(args.ckpt) - 1, T)
tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

alphas = np.linspace(0.0, 1.0, args.num_points)

for path in args.ckpt:
    print(path)

step = 0
for i in range(len(args.ckpt) - 1):
    base_model.load_state_dict(torch.load(args.ckpt[i])['model_state'])
    w_1 = get_weights(base_model)

    base_model.load_state_dict(torch.load(args.ckpt[i + 1])['model_state'])
    w_2 = get_weights(base_model)
    for alpha in alphas[1 if i > 0 else 0:]:
        w = (1.0 - alpha) * w_1 + alpha * w_2
        offset = 0
        for parameter in base_model.parameters():
            size = np.prod(parameter.size())
            value = w[offset:offset+size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        utils.update_bn(loaders['train'], base_model)

        tr_res = utils.test(loaders['train'], base_model, criterion, regularizer)
        te_res = utils.test(loaders['test'], base_model, criterion, regularizer)

        tr_loss[step] = tr_res['loss']
        tr_nll[step] = tr_res['nll']
        tr_acc[step] = tr_res['accuracy']
        tr_err[step] = 100.0 - tr_acc[step]
        te_loss[step] = te_res['loss']
        te_nll[step] = te_res['nll']
        te_acc[step] = te_res['accuracy']
        te_err[step] = 100.0 - te_acc[step]

        values = [ts[step], tr_loss[step], tr_nll[step], tr_err[step], te_nll[step], te_err[step]]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if step % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        step += 1


np.savez(
    os.path.join(args.dir, 'chain.npz'),
    ts=ts,
    tr_loss=tr_loss,
    tr_nll=tr_nll,
    tr_acc=tr_acc,
    tr_err=tr_err,
    te_loss=te_loss,
    te_nll=te_nll,
    te_acc=te_acc,
    te_err=te_err,
)
