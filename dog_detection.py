import logging

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models
import numpy as np
import cv2

from tools.snippets import (quick_log_setup, mkdir)
from tools.voc import (VOC_ocv, transforms_voc_ocv_eval,
                       sequence_batch_collate_v2, enforce_all_seeds)
from tools.dogs import (
        eval_stats_at_threshold, read_metadata,
        produce_gt_dog_boxes, produce_fake_centered_dog_boxes,
        visualize_dog_boxes)


def dog_detection():
    """False
    实现两个简单的狗检测基线：
      A. 在每张图像中心预测一个固定大小的框（面积占50%）
      B. 偷看真实标注，在有狗的图像中预测一个中心框
    同时提供评估和可视化功能。
    """
    # 配置部分：加载数据集和元数据
    voc_folder = 'voc_dataset'  # VOC数据集存储路径
    dataset_test = VOC_ocv(
        voc_folder, year='2007', image_set='test',
        download=False, transforms=transforms_voc_ocv_eval
    )
    metadata_test = read_metadata(dataset_test)  # 读取元数据
    metadata_test = dict(list(metadata_test.items())[:500])  # 只保留前500张图像
    all_gt_dogs = produce_gt_dog_boxes(metadata_test)  # 生成真实标注的狗框

    # 基线A：在每张图像中心预测一个固定框
    all_centerbox_dogs = produce_fake_centered_dog_boxes(metadata_test, scale=0.3)
    stats_df = eval_stats_at_threshold(all_centerbox_dogs, all_gt_dogs)
    log.info('A. ALL CENTERBOX dogs\n{}'.format(stats_df))

    # 基线B：偷看真实标注，仅在有狗的图像中预测中心框
    all_cheating_centerbox_dogs = {
        k: v for k, v in all_centerbox_dogs.items() if len(all_gt_dogs[k])
    }
    stats_df = eval_stats_at_threshold(all_cheating_centerbox_dogs, all_gt_dogs)
    log.info('B. CHEATING CENTERBOX dogs\n{}'.format(stats_df))

    # 基线C：结合分类器预测得分，为中心框分配分数
    initial_seed = 42  # 设置随机种子
    num_workers = 0  # 设置为0以便调试
    inputs_ckpt = "./model_at_epoch_019.pth.tar"  # 预训练模型路径
    subset_test = torch.utils.data.Subset(dataset_test, list(range(500)))  # 只保留前500张图像
    dataloader_test = torch.utils.data.DataLoader(
        subset_test, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=sequence_batch_collate_v2
    )
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)  # 输出层改为20类
    model.to(device)
    model.eval()
    states = torch.load(inputs_ckpt, map_location=device)
    model.load_state_dict(states["model_sdict"])
    enforce_all_seeds(initial_seed)  # 固定随机种子

    # 定位ResNet50的最后一个卷积层
    target_layer = model.layer4[-1].conv3
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # 注册钩子
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)


    # 进行预测并为中心框分配分数
    all_scored_centerdogs = {}
    for i_batch, (data, target, meta) in enumerate(tqdm(dataloader_test, "评估测试集")):
        imname = meta[0]['xml_parsed']['annotation']['filename']
        data = data.to(device)
        
        # 前向传播
        pred = torch.sigmoid(model(data))
        
        # 反向传播计算梯度（针对狗类别，索引为4）
        model.zero_grad()
        pred[:,4].backward(retain_graph=True)
        
        # 生成热力图
        weights = torch.mean(gradients["value"], dim=(2,3), keepdim=True)
        cam = torch.sum(weights * activations["value"], dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=data.shape[2:], mode="bilinear")
        heatmap = cam.squeeze().cpu().numpy()
        
        # 归一化到0-1
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # 阈值化和轮廓检测（阈值设为0.5）
        _, binary_map = cv2.threshold(heatmap, 0.5, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 生成边界框
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # 过滤小轮廓
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x+w, y+h, pred[0,4].item()])  # [xmin, ymin, xmax, ymax, 分数]
        
        # 保存结果
        all_scored_centerdogs[imname] = np.array(boxes) if boxes else np.empty((0,5))
    stats_df = eval_stats_at_threshold(all_scored_centerdogs, all_gt_dogs)
    log.info('C. 得分中心框检测结果：\n{}'.format(stats_df))

    # 可视化结果
    fold = mkdir('visualize/scored_centerbox_dogs')
    visualize_dog_boxes(fold, all_scored_centerdogs, all_gt_dogs, metadata_test)


if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    dog_detection()
