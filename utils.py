import numpy as np
import torch
import torch.nn as nn
from argument_parser import parser_train
import torch.nn.functional as F

args = parser_train()

if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
elif args.dataset == 'cifar100':
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
elif args.dataset == 'svhn':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
mu = torch.tensor(mean).view(3, 1, 1).to(device)
std = torch.tensor(std).view(3, 1, 1).to(device)
epsilon = args.epsilon
restarts = args.restarts
pgd_iter = args.pgd_iter
pgd_step = args.pgd_step


def normalize(x):
    return (x - mu) / std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, norm="l_inf"):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        delta = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(pgd_iter):
            output = model(normalize(X + delta))
            loss = F.cross_entropy(output, y)
            loss.backward()
            g = delta.grad.detach()
            d = delta.clone()
            if norm == "l_inf":
                d = torch.clamp(d + pgd_step * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * pgd_step).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            delta.data = clamp(d, -X, 1 - X)
            delta.grad.zero_()
        output = model(normalize(X + delta))
        criterion = nn.CrossEntropyLoss(reduction='none')
        all_loss = criterion(output, y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return delta


def attack_pgd_maxup(model, X, y, norm="l_inf"):
    """
    X: (batch_size, args.m, C, H, W)
    y: (batch_size, args.m, num_classes)
    """
    max_loss = torch.zeros(y.size(0), y.size(1)).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0) * delta.size(1), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0) * delta.size(1), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1) # B * M, 1, 1, 1
            delta = (delta.reshape(-1, 3, X.shape[-1], X.shape[-1]) * r / n * epsilon).reshape(X.shape)
        delta = clamp(delta, -X, 1 - X)
        delta.requires_grad_(True)
        for _ in range(pgd_iter):
            imgs = normalize(X + delta)
            imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], 3, X.shape[-1], X.shape[-1]))
            output = model(imgs)  # return (B * M, num_classes)
            output = output.reshape(y.shape)
            loss = torch.sum(-F.log_softmax(output, dim=-1) * y, -1)
            loss.backward(torch.ones_like(loss))
            g = delta.grad.detach()
            d = delta.clone()
            if norm == "l_inf":
                d = torch.clamp(d + pgd_step * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0] * g.shape[1], -1), dim=1).view(-1, 1, 1, 1)  # (B*M, 1, 1, 1)
                scaled_g = g.reshape((-1, 3, g.shape[-1], g.shape[-1])) / (g_norm + 1e-10)  # (B*M, C, H, W)
                d = (d.reshape((-1, 3, d.size(-1), d.size(-1))) + scaled_g * pgd_step).view(d.size(0) * d.size(1), -1).renorm(p=2, dim=0, maxnorm=epsilon).reshape(X.shape)
            delta.data = clamp(d, -X, 1 - X)
            delta.grad.zero_()
        imgs = normalize(X + delta)
        imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], 3, X.shape[-1], X.shape[-1]))
        output = model(imgs)
        output = output.reshape(y.shape)
        all_loss = torch.sum(-F.log_softmax(output, dim=-1) * y, -1)  # B, M
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return delta


def valid(test_data, model):
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(test_data):
        x, y = batch
        x, y = x.to(device), y.to(device)
        if args.train == 'adversarial':
            delta = attack_pgd(model, x, y)
            delta = delta.detach()
        else:
            delta = torch.zeros_like(x)
        robust_output = model(normalize(torch.clamp(x + delta[:x.size(0)], min=0, max=1)))
        robust_loss = criterion(robust_output, y)
        output = model(normalize(x))
        loss = criterion(output, y)
        test_robust_loss += robust_loss.item() * y.size(0)
        test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)
    return test_robust_loss, test_robust_acc, test_loss, test_acc, test_n
