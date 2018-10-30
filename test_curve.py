import argparse
import torch

import curves
import data
import models


parser = argparse.ArgumentParser(description='Test DNN curve')

parser.add_argument('--dataset', type=str, default=None, metavar='DATASET',
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

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE', required=True,
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset is not None:
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )
    loader = loaders['test']
else:
    num_classes = 10
    loader = [(torch.randn((args.batch_size, 3, 32, 32)), None) for i in range(20)]

architecture = getattr(models, args.model)

curve = getattr(curves, args.curve)
curve_model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    True,
    True,
    architecture_kwargs=architecture.kwargs,
)

base = [architecture.base(num_classes, **architecture.kwargs) for _ in range(2)]
for base_model, path, k in zip(base, [args.init_start, args.init_end], [0, args.num_bends - 1]):
    if path is not None:
        checkpoint = torch.load(path)
        print('Loading %s as point #%d' % (path, k))
        base_model.load_state_dict(checkpoint['model_state'])
    curve_model.import_base_parameters(base_model, k)

if args.init_linear:
    print('Linear initialization.')
    curve_model.init_linear()
curve_model.cuda()
for base_model in base:
    base_model.cuda()

t = torch.FloatTensor([0.0]).cuda()
for base_model, t_value in zip(base, [0.0, 1.0]):
    print('T: %f' % t_value)
    t.data.fill_(t_value)
    curve_model.import_base_buffers(base_model)
    curve_model.eval()
    base_model.eval()

    max_error = 0.0
    for i, (input, _) in enumerate(loader):
        input = input.cuda(async=True)

        base_ouput = base_model(input)
        curve_output = curve_model(input, t)

        error = torch.max(torch.abs(base_ouput - curve_output)).item()
        print('Batch #%d. Error: %g' % (i, error))
        max_error = max(max_error, error)
    print('Max error: %g' % max_error)
    assert max_error < 1e-4, 'Error is too big (%g)' % max_error
