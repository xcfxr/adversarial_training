import numpy as np
import torch
import random

from torch.utils.data.dataset import Dataset


class Mixup(Dataset):
    def __init__(self, dataset, m=1, beta=1.0, num_classes=10):
        self.dataset = dataset
        self.m = m
        self.beta = beta
        self.num_classes = num_classes

    def __getitem__(self, index):
        m_imgs = []
        m_lbls = []
        for _ in range(self.m):
            img, lbl1 = self.dataset[index]
            lbl1 = self.one_hot(lbl1)
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))
            img2, lbl2 = self.dataset[rand_index]
            lbl2 = self.one_hot(lbl2)
            img = lam * img + (1. - lam) * img2
            lbl = lbl1 * lam + lbl2 * (1. - lam)
            m_imgs.append(img)
            m_lbls.append(lbl)
        m_imgs = torch.stack(m_imgs, 0)
        m_lbls = torch.stack(m_lbls, 0)
        return m_imgs, m_lbls

    def __len__(self):
        return len(self.dataset)

    def one_hot(self, idx):
        lb = torch.zeros(self.num_classes, dtype=torch.float32)
        lb[idx] = 1
        return lb
