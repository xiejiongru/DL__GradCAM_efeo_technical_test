import re
import collections
import logging
import xml.etree.ElementTree as ET
import random
from pathlib import Path

import numpy as np

import cv2

import torch
import torch.autograd
import torchvision.models
from torch.utils.data.dataloader import default_collate

import albumentations.pytorch
import albumentations as A

log = logging.getLogger(__name__)


# / Pytorch training boilerplate


def enforce_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# Custom collate function to pass metainformation
def sequence_batch_collate_v2(batch):
    assert isinstance(batch[0], collections.abc.Sequence), \
            'Only sequences supported'
    # From gunnar code
    transposed = zip(*batch)
    collated = []
    for samples in transposed:
        if isinstance(samples[0], collections.abc.Mapping) \
               and 'do_not_collate' in samples[0]:
            c_samples = samples
        elif getattr(samples[0], 'do_not_collate', False) is True:
            c_samples = samples
        else:
            c_samples = default_collate(samples)
        collated.append(c_samples)
    return collated


def construct_optimizer(model,
        optimizing_method='adamw',
        base_lr=0.00005, momentum=0.9, dampening=0.0,
        nesterov=False, weight_decay=0):
    # slowfast/models/optimizer.py
    # BN params should not get weight decayed
    bn_params = []
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    optim_params = [
        {"params": bn_params, "weight_decay": 0},
        {"params": non_bn_parameters, "weight_decay": weight_decay},
    ]
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )
    optim_params = model.parameters()
    if optimizing_method == 'sgd':
        optimizer = torch.optim.SGD(optim_params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov)
    elif optimizing_method == 'adamw':
        optimizer = torch.optim.AdamW(optim_params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay)
    else:
        raise NotImplementedError()
    return optimizer


class Manager_checkpoint_name(object):
    ckpt_re = r'model_at_epoch_(?P<i_epoch>\d*).pth.tar'
    ckpt_format = 'model_at_epoch_{:03d}.pth.tar'

    @classmethod
    def get_checkpoint_path(self, rundir, i_epoch) -> Path:
        save_filepath = rundir/self.ckpt_format.format(i_epoch)
        return save_filepath

    @classmethod
    def find_checkpoints(self, rundir):
        checkpoints = {}
        for subfolder_item in rundir.iterdir():
            search = re.search(self.ckpt_re, subfolder_item.name)
            if search:
                i_epoch = int(search.groupdict()['i_epoch'])
                checkpoints[i_epoch] = subfolder_item
        return checkpoints

    @classmethod
    def find_last_checkpoint(self, rundir):
        checkpoints = self.find_checkpoints(rundir)
        if len(checkpoints):
            checkpoint_path = max(checkpoints.items())[1]
        else:
            checkpoint_path = None
        return checkpoint_path


class Checkpointer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def save_epoch(self, rundir, i_epoch):
        # model_{epoch} - "after epoch was finished"
        save_filepath = (Manager_checkpoint_name
                .get_checkpoint_path(rundir, i_epoch))
        states = {
            'i_epoch': i_epoch,
            'model_sdict': self.model.state_dict(),
            'optimizer_sdict': self.optimizer.state_dict(),
        }
        torch.save(states, str(save_filepath))
        log.debug(f'Saved model. Epoch {i_epoch}')
        log.debug('Saved to {}'.format(save_filepath))

    def restore_model_magic(
            self, checkpoint_path,
            starting_model=None, training_start_epoch=0):
        if checkpoint_path is not None:
            # Continue training
            states = torch.load(checkpoint_path)
            self.model.load_state_dict(states['model_sdict'])
            self.optimizer.load_state_dict(states['optimizer_sdict'])
            start_epoch = states['i_epoch']
            start_epoch += 1
            log.info('Continuing training from checkpoint {}. '
                    'Epoch {} (ckpt + 1)'.format(checkpoint_path, start_epoch))
        else:
            start_epoch = training_start_epoch
            if starting_model is not None:
                states = torch.load(starting_model)
                self.model.load_state_dict(states['model_sdict'])
                log.info(('Starting new training, '
                    'initialized from model {}, at epoch {}').format(
                        starting_model, start_epoch))
            else:
                log.info(('Starting new training, '
                    'empty model, at epoch {}').format(start_epoch))
        return start_epoch


def qacc_sigmoid(output, Y):
    pred = (torch.sigmoid(output) > 0.5)
    return pred.eq(Y).sum().item()/len(Y)


def lr_func_steps_with_relative_lrs(
        solver_steps, MAX_EPOCH, solver_lrs, base_lr, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    """
    steps = solver_steps + [MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    ind = ind - 1
    return solver_lrs[ind] * base_lr


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# / Dataset related functions


VOC_CLASSES = [
        'person', 'bird', 'cat', 'cow', 'dog',
        'horse', 'sheep', 'aeroplane', 'bicycle',
        'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant',
        'sofa', 'tvmonitor']


# / Preprocessing transforms
S = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
anormalize = A.Normalize(mean=MEAN, std=STD)
# // During training - apply random augmentations
atransform_train = A.Compose([
    A.RandomResizedCrop(
        (S, S), scale=(0.765, 1.0), ratio=(1.0, 1.0)),
    A.HorizontalFlip(),
    anormalize,
    albumentations.pytorch.ToTensorV2(),
])
# // During evaluation - simply resize
atransform_eval = A.Compose([
    A.Resize(height=S, width=S),
    anormalize,
    albumentations.pytorch.ToTensorV2(),
])


def create_multilabel_target(target):
    target_vector = np.zeros(len(VOC_CLASSES))
    objects = target['annotation']['object']
    for obj in objects:
        ind = VOC_CLASSES.index(obj['name'])
        target_vector[ind] = 1
    target_vector_t = torch.from_numpy(target_vector)
    return target_vector_t


def transforms_voc_ocv_train(image, target):
    im_torch = atransform_train(image=image)['image']
    target_vector_t = create_multilabel_target(target)
    return im_torch, target_vector_t


def transforms_voc_ocv_eval(image, target):
    im_torch = atransform_eval(image=image)['image']
    target_vector_t = create_multilabel_target(target)
    return im_torch, target_vector_t


# Subclassing torchvision dataset to load images with our transforms
class VOC_ocv(torchvision.datasets.VOCDetection):
    def __init__(
            self, root, year, image_set, download, transforms,
            n_retries=3, pass_image=False):
        super(VOC_ocv, self).__init__(
                root, year, image_set,
                download, transforms=transforms)
        self.n_retries = n_retries
        self.pass_image = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        # Load basic image with opencv
        impath = self.images[index]
        img = cv2.imread(str(impath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Load xml annotation
        xml_parsed = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        # Apply transform
        im_torch = None
        for i_try in range(self.n_retries):
            try:
                im_torch, target = self.transforms(img, xml_parsed)
                break
            except Exception as e:
                log.info(f"Failed to load transform for {impath} (try {i_try}")
                log.exception(e)
        if im_torch is None:
            raise RuntimeError(f"Failed to load {impath} after {i_try} tries")
        meta = {'impath': impath,
                'xml_parsed': xml_parsed,
                'do_not_collate': True}
        # Pass full image if the flag is set
        if self.pass_image:
            meta['image'] = img
        return im_torch, target, meta
