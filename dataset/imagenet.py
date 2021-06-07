"""
get data loaders
"""
from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, distributed
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import torch
from PIL import ImageFilter
import random



class CropsTransform:
    """Take four random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        img0 = self.train_transform(x)
        img1 = self.base_transform(x)
        img2 = self.base_transform(x)
        img3 = self.base_transform(x)
        img = torch.stack([img0,img1,img2,img3])
        return img


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets/imagenet'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data/imagenet'
    else:
        data_folder = '../../datasets/Imagenet/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class AugmentImageFolder(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if np.random.rand() < 0.5:
            img = np.array(img)
            img = img[:,::-1,:]

        img0 = np.rot90(img, 0).copy()
        img0 = Image.fromarray(img0)
        img0 = self.transform(img0)

        img1 = np.rot90(img, 1).copy()
        img1 = Image.fromarray(img1)
        img1 = self.transform(img1)

        img2 = np.rot90(img, 2).copy()
        img2 = Image.fromarray(img2)
        img2 = self.transform(img2)

        img3 = np.rot90(img, 3).copy()
        img3 = Image.fromarray(img3)
        img3 = self.transform(img3)

        img = torch.stack([img0,img1,img2,img3])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_test_loader(dataset='imagenet', batch_size=128, num_workers=8):
    """get the test data loader"""

    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(data_folder, 'val')
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_dataloader_sample(dataset='imagenet', batch_size=128, num_workers=8, is_sample=False, k=4096):
    """Data Loader for ImageNet"""

    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # add data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    train_set = ImageFolderSample(train_folder, transform=train_transform, is_sample=is_sample, k=k)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    print('num_samples', len(train_set.samples))
    print('num_class', len(train_set.classes))

    return train_loader, test_loader, len(train_set), len(train_set.classes)


def get_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=16):
    """
    Data Loader for imagenet
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
 
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    #train_set = datasets.ImageFolder(
    #    train_folder,
    #    CropsTransform(transforms.Compose(augmentation)))
    
    train_set = AugmentImageFolder(train_folder, transform=train_transform)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    return train_set, test_set

