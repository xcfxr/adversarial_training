import logging
import os.path
import time
from resnet import PreActResNet18
from load_data import loader_maxup
from utils import *
from maxup_loss import MaxupCrossEntropyLoss
from wideresnet import WideResNet


def main():
    log_dir = os.path.join('adversarial_experiment', f'{args.norm}_{args.dataset}_{args.augment}_m{args.m}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=10,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'output.log')),
            logging.StreamHandler(),
        ])
    logger.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    epochs = args.epochs

    def lr_schedule(t):
        if t / args.epochs < 0.5:
            return 0.1
        elif t / args.epochs < 0.75:
            return 0.01
        else:
            return 0.001

    best_robust_acc = 0
    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc'
        ' \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

    train_data, test_data, num_classes = loader_maxup(args.data_dir, args.dataset, args.augment, args.batch_size,
                                                      args.num_workers,
                                                      args.m)
    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_classes).to(device)
    else:
        model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        criterion = MaxupCrossEntropyLoss(args.m)
        for i, batch in enumerate(train_data):
            X, y = batch
            X, y = X.to(device), y.to(device)
            lr = lr_schedule(epoch + (i + 1) / len(train_data))
            opt.param_groups[0].update(lr=lr)
            delta = attack_pgd_maxup(model, X, y, args.norm)  # delta
            imgs = normalize(torch.clamp(X + delta[:X.size(0)], min=0, max=1))
            imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1], 3, imgs.shape[-1], imgs.shape[-1])
            robust_output = model(imgs)
            robust_loss = criterion(robust_output, y, args.use_all)
            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            output = model(normalize(X).reshape(X.size(0) * args.m, 3, X.size(-1), X.size(-1)))
            loss = criterion(output, y, args.use_all)
            train_robust_loss += robust_loss.item() * y.size(0) * y.size(1)
            train_robust_acc += (robust_output.max(1)[1] == y.reshape(robust_output.shape).max(1)[1]).sum().item()
            train_loss += loss.item() * y.size(0) * y.size(1)
            train_acc += (output.max(1)[1] == y.reshape(output.shape).max(1)[1]).sum().item()
            train_n += y.size(0) * y.size(1)

        train_time = time.time()
        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        criterion = nn.CrossEntropyLoss()
        for i, batch in enumerate(test_data):
            X, y = batch
            X, y = X.to(device), y.to(device)
            if args.train == 'vanilla':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, args.norm)
                delta = delta.detach()
            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=0, max=1)))
            robust_loss = criterion(robust_output, y)
            output = model(normalize(X))
            loss = criterion(output, y)
            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
        test_time = time.time()
        logger.info(
            '%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
            epoch, train_time - start_time, test_time - train_time, lr,
                   train_loss / train_n, train_acc / train_n, train_robust_loss / train_n, train_robust_acc / train_n,
                   test_loss / test_n, test_acc / test_n, test_robust_loss / test_n, test_robust_acc / test_n)
        if test_robust_acc / test_n > best_robust_acc:
            torch.save({
                'state_dict': model.state_dict(),
                'test_robust_acc': test_robust_acc / test_n,
                'test_robust_loss': test_robust_loss / test_n,
                'test_loss': test_loss / test_n,
                'test_acc': test_acc / test_n,
            }, os.path.join(log_dir, 'model_best.pth'))
            best_robust_acc = test_robust_acc / test_n


if __name__ == "__main__":
    main()
