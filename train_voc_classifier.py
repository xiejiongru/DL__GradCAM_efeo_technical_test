import logging

import sklearn.metrics
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models

from tools.snippets import (quick_log_setup, mkdir, Averager)
from tools.voc import (
        VOC_CLASSES, VOC_ocv, enforce_all_seeds,
        transforms_voc_ocv_train, transforms_voc_ocv_eval,
        sequence_batch_collate_v2, construct_optimizer,
        Checkpointer, Manager_checkpoint_name,
        lr_func_steps_with_relative_lrs, set_lr, qacc_sigmoid)

log = logging.getLogger(__name__)


def train_voc_classifier():
    """
    此函数将在VOC2007训练集上微调一个简单的多标签分类器，
    使用预训练的Imagenet resnet50作为起点
    """
    # / 配置
    initial_seed = 42
    num_workers = 4
    voc_folder = 'voc_dataset'
    # Directory to save outputs to
    rundir = mkdir('training_outputs')
    # // Training configurations
    MAX_EPOCH = 20
    solver_steps = [0, 5, 15]
    solver_lrs = [1, 0.1, 0.01, 0.001]
    base_lr = 0.00005

    enforce_all_seeds(initial_seed)

    dataset_train = VOC_ocv(
            voc_folder, year='2007', image_set='train',
            download=True, transforms=transforms_voc_ocv_train)
    dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=32,
            shuffle=True, num_workers=num_workers,
            collate_fn=sequence_batch_collate_v2)

    dataset_test = VOC_ocv(
            voc_folder, year='2007', image_set='test',
            download=True, transforms=transforms_voc_ocv_eval)
    dataloader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=32,
            shuffle=False, num_workers=num_workers,
            collate_fn=sequence_batch_collate_v2)

    # Define resnet50 model, load the imagenet weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_classes = 20
    model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    model.to(device)

    optimizer = construct_optimizer(model, base_lr=base_lr)
    loss_criterion = nn.BCEWithLogitsLoss()

    ckpt = Checkpointer(model, optimizer)
    checkpoint_path = (Manager_checkpoint_name
            .find_last_checkpoint(rundir))

    avg_loss = Averager()
    avg_acc = Averager()

    start_epoch = ckpt.restore_model_magic(checkpoint_path)

    # Train loop
    for i_epoch in range(start_epoch, MAX_EPOCH):
        epoch_seed = i_epoch + initial_seed
        enforce_all_seeds(epoch_seed)

        model.train()
        for i_batch, (data, target, meta) in enumerate(dataloader_train):
            data, target = map(lambda x: x.to(device), (data, target))

            # Set appropriate learning rate
            total_batches = len(dataloader_train)
            lr = lr_func_steps_with_relative_lrs(
                    solver_steps, MAX_EPOCH, solver_lrs, base_lr,
                    i_epoch + float(i_batch)/total_batches)
            set_lr(optimizer, lr)

            # Output
            output = model(data)
            loss = loss_criterion(output, target)

            # Measure params
            with torch.no_grad():
                acc = qacc_sigmoid(output, target)
            avg_acc.update(acc)
            avg_loss.update(loss.data.item(), target.size(0))

            # Gradient and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log train stats
            if i_batch % 25 == 0:
                Nb = len(dataloader_train)
                loss_str = (f'loss(all/last):{avg_loss.avg:.4f}/{avg_loss.last:.4f}')
                acc_str = (f'acc(all/last):{avg_acc.avg:.2f}/{avg_acc.last:.2f}')
                log.info(f'i_epoch={i_epoch}, i_batch={i_batch}/{Nb}; '
                        f'lr={lr}; TRAIN: {loss_str} {acc_str}')

        ckpt.save_epoch(rundir, i_epoch)
        model.eval()

        targets = []
        outputs = []
        for i_batch, (data, target, meta) in enumerate(dataloader_test):
            data, target = map(lambda x: x.to(device), (data, target))
            with torch.no_grad():
                output = model(data)
                output_sigm = torch.sigmoid(output)
                output_np = output_sigm.detach().cpu().numpy()
                targets.append(target.cpu())
                outputs.append(output_np)
        targets = np.vstack(targets)
        outputs = np.vstack(outputs)

        aps = {}
        for label, X, Y in zip(VOC_CLASSES, outputs.T, targets.T):
            aps[label] = sklearn.metrics.average_precision_score(Y, X)
        aps['MEAN'] = np.mean(list(aps.values()))
        s = pd.Series(aps)*100
        log.info('epoch {} mAP: {}'.format(i_epoch, s['MEAN']))


if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    train_voc_classifier()
