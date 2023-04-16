import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

import numpy as np
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


        
def save_checkpoint(filepath, obj):
    logging.info("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    logging.info("Complete.")
    
    
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    logging.info("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    logging.info("Complete.")
    return checkpoint_dict


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    logging.info("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    logging.info("Complete.")
    return checkpoint_dict

        
def pad_or_trim(array, length: int, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        
        mask = torch.ones_like(array)
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
            mask = mask.index_select(dim=axis, index=torch.arange(length, device=array.device))
            
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
            
            mask = F.pad(mask, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        mask = np.ones_like(array)
        if array.shape[axis] > length:
            
            array = array.take(indices=range(length), axis=axis)
            mask = mask.take(indices=range(length), axis=axis)
            
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
            mask = np.pad(mask, pad_widths)

    return array, mask


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger