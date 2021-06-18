import os
import sys
import importlib
import numpy as np
# import torch

import logging
# logging.basicConfig(level=logging.DEBUG)

class Formatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "[I] %(message)s"
        elif record.levelno == logging.DEBUG:
            self._style._fmt = "[D] %(message)s"
        elif record.levelno == logging.ERROR:
            self._style._fmt = "[E] %(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)

log = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(Formatter())
log.setLevel(logging.INFO)
log.addHandler(handler)


def DebugOn():
    log.setLevel(logging.DEBUG)

def D(msg, *args, **kwargs):
    log.debug(msg, *args, **kwargs)


def I(msg, *args, **kwargs):
    log.info(msg, *args, **kwargs)


def E(msg, *args, **kwargs):
    log.error(msg, *args, **kwargs)



def load_python_source(path_src):
    path_src  = os.path.abspath(path_src)
    dir_src   = os.path.dirname(path_src)
    bname_src = os.path.basename(path_src)

    if dir_src not in sys.path:
        sys.path.append(dir_src)

    mod_src = importlib.import_module(bname_src[:-3])
    importlib.reload(mod_src)   # reload is important
    return mod_src


def ignore_numpy_invalid():
    np.seterr(invalid='ignore')


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



def cuda_mem_usage():
    print(torch.cuda.get_device_name(0))
    print('CUDA Memory Usage:')
    # print('CUDA Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('CUDA Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('CUDA Allocated:', round(torch.cuda.memory_allocated(0)/1024,1), 'KB')
    print('CUDA Cached:   ', round(torch.cuda.memory_reserved(0)/1024,1), 'KB')
