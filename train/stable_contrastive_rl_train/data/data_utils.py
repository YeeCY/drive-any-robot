import numpy as np
import os
from PIL import Image
from typing import Any, Iterable

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class GeometricClassBalancer:
    def __init__(self, classes: Iterable, discount: float) -> None:
        """
        A class balancer that will sample classes geometrically, but will prioritize classes that have been sampled less

        Args:
            classes (Iterable): The classes to balance
        """
        # self.counts = {}
        # for c in classes:
        #     self.counts[c] = 0
        self.classes = classes
        self.discount = discount

    def sample(self, class_filter_func=None) -> Any:
        """
        # Sample the softmax of the negative logits to prioritize classes that have been sampled less

        Args:
            class_filter_func (Callable, optional): A function that takes in a class and returns a boolean. Defaults to None.
        """
        if class_filter_func is None:
            valid_classes = self.classes
        else:
            # keys = [k for k in self.counts.keys() if class_filter_func(k)]
            valid_classes = [c for c in self.classes if class_filter_func(c)]
        if len(valid_classes) == 0:
            return None  # no valid classes to sample

        # TODO (chongyiz): do we need to shift this geometric distribution one timestep into the future?
        probs = self.discount ** (np.asarray(valid_classes) - np.min(valid_classes))
        probs = probs / np.sum(probs, keepdims=True)
        class_choice = np.random.choice(valid_classes, p=probs)

        # values = [-(self.counts[k] - min(self.counts.values())) for k in keys]
        # p = F.softmax(torch.Tensor(values), dim=0).detach().cpu().numpy()
        # class_index = np.random.choice(list(range(len(keys))), p=p)
        # class_choice = keys[class_index]
        # self.counts[class_choice] += 1
        return class_choice

    def __str__(self) -> str:
        string = ""
        for c in self.counts:
            string += f"{c}: {self.counts[c]}\n"
        return string
