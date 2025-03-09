import logging

import sklearn.metrics
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision.models

from tools.snippets import quick_log_setup
from tools.voc import (
        VOC_CLASSES, VOC_ocv, enforce_all_seeds,
        transforms_voc_ocv_eval, sequence_batch_collate_v2)

log = logging.getLogger(__name__)


def evaluate_voc_classifier():
    """
    评估VOC2007测试集上的多标签分类器性能。
    主要步骤包括：
      1. 加载预训练模型和测试数据
      2. 对测试集进行预测
      3. 计算每个类别的平均精度（AP）和整体均值AP
    """
    # 配置部分：设置随机种子、加载数据集、定义模型等
    initial_seed = 42  # 设置初始随机种子以确保可重复性
    num_workers = 4  # 数据加载的线程数
    voc_folder = 'voc_dataset'  # VOC数据集存储路径
    inputs_ckpt = 'model_at_epoch_019.pth.tar'  # 预训练模型路径

    # 加载测试数据集
    dataset_test = VOC_ocv(
        voc_folder, year='2007', image_set='test',
        download=True, transforms=transforms_voc_ocv_eval
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32,
        shuffle=False, num_workers=num_workers,
        collate_fn=sequence_batch_collate_v2
    )

    # 定义ResNet50模型并加载预训练权重
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)  # 输出层改为20类
    model.to(device)
    model.eval()

    # 固定随机种子
    enforce_all_seeds(initial_seed)

    # 加载微调后的模型权重
    states = torch.load(inputs_ckpt, map_location=device)
    model.load_state_dict(states['model_sdict'])

    # 进行预测
    targets = []
    outputs = []
    for i_batch, (data, target, meta) in enumerate(tqdm(dataloader_test)):
        data, target = map(lambda x: x.to(device), (data, target))
        with torch.no_grad():
            output = model(data)
            output_sigm = torch.sigmoid(output)  # 使用sigmoid激活函数
            output_np = output_sigm.detach().cpu().numpy()
            targets.append(target.cpu())
            outputs.append(output_np)
    targets = np.vstack(targets)
    outputs = np.vstack(outputs)

    # 计算每个类别的平均精度（AP）
    aps = {}
    for label, X, Y in zip(VOC_CLASSES, outputs.T, targets.T):
        aps[label] = sklearn.metrics.average_precision_score(Y, X)
    aps['MEAN'] = np.mean(list(aps.values()))  # 计算均值AP
    s = pd.Series(aps) * 100  # 转换为百分比形式
    log.info('多标签分类性能（AP）：\n{}'.format(s))


if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    evaluate_voc_classifier()
