import random
import time

import numpy as np
import torch


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Timer:
    """Context manager for timing a block of code.

    Usage:
        with Timer() as t:
            ...
        print(t.elapsed)
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


class AverageMeter:
    """Tracks running average of a metric (loss, accuracy, ...)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location=None) -> torch.nn.Module:
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model
