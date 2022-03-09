import torch
from torch.utils.data import Dataset


class Maxup(Dataset):
    def __init__(self, dataset, m=1, theta=0.5, num_classes=10):
        self.dataset = dataset
        self.m = m
        self.theta = theta
        self.num_classes = num_classes

    def __getitem__(self, index):
        m_imgs = []
        m_lbls = []
        for _ in range(self.m):
            img, lbl1 = self.dataset[index]
            img2 = torch.normal(mean=img, std=torch.ones_like(img) * self.theta)
            lbl = self.one_hot(lbl1)
            m_imgs.append(img2)
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
