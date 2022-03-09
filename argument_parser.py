import argparse


def parser_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=[
        'cifar10', 'cifar100', 'svhn'
    ])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epsilon', default=8/255, type=float)
    parser.add_argument('--pgd-iter', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-step', default=2/255, type=float)
    parser.add_argument('--attack', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--seed', type=int, default=202201)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--augment', type=str, default='cutmixup', choices=[
        'none', 'mixup', 'cutmix', 'cutmixup']
    )
    parser.add_argument('--norm', type=str, default='l_2', choices=[
        'l_inf', 'l_2'
    ])
    parser.add_argument('--use-all', type=bool, default=False)
    parser.add_argument('--m', type=int, default=2, help='number of maxup')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train', type=str, default='attack', choices=['vanilla', 'attack'])
    parser.add_argument('--alpha', type=float, default=1.0, help='parameter of argument')
    return parser.parse_args()
