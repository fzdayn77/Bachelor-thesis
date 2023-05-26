#
# This source code is inspired from Pytorch Lightning SimCLR implementation :
#   https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
#

import cv2
import numpy as np

import torch
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms

from typing import Optional


class GaussianBlur(object):
    """
    Blurs the given image with separable convolution as described in the SimCLR paper(ArXiv, https://arxiv.org/abs/2002.05709).
    """

    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        # less than 50%
        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class simCLR_training_data_augmentation(object):
    """
    Implementation of the data augmentations on the trainig data
    proposed in the SimCLR official paper
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = False,
        jitter_strength: float = 1.,
        normalize: Optional[transforms.Normalize] = None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, # type: ignore
            0.8 * self.jitter_strength, #type: ignore
            0.8 * self.jitter_strength, #type: ignore
            0.2 * self.jitter_strength  #type: ignore
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.input_height), p=0.5))

        data_transforms.append(transforms.ToTensor())

        if self.normalize:
            data_transforms.append(normalize)

        # Transformations on the training data
        self.train_transform = transforms.Compose(data_transforms)

    def __call__(self, z):
        transform = self.train_transform

        z_i = transform(z)
        z_j = transform(z)

        return z_i, z_j


class simCLR_eval_data_augmentation(object):
    """
    Implementation of the data augmentations on the testing data
    proposed in the SimCLR official paper
    """

    def __init__(
        self,
        input_height: int = 224,
        crop: bool = False,
        normalize: Optional[transforms.Normalize] = None
    ):
        self.input_height = input_height
        self.normalize = normalize

        data_transforms = [
            transforms.Resize(self.input_height),
            transforms.ToTensor()
        ]

        if crop:
            data_transforms.append(transforms.RandomResizedCrop(size=self.input_height))

        if self.normalize:
            data_transforms.append(normalize)

        # Transformations on the testing data
        self.test_transform = transforms.Compose(data_transforms)

    def __call__(self, z):
        transform = self.test_transform

        z_i = transform(z)
        z_j = transform(z)

        return z_i, z_j