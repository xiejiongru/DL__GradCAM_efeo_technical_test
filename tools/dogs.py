import logging
import xml.etree.ElementTree as ET

import cv2  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from pathlib import Path
from .snippets import (visualize_bbox)

log = logging.getLogger(__name__)


def voc_ap(rec, prec, use_07_metric=False):
    """计算给定精确率和召回率的VOC AP（平均精度）。
    如果use_07_metric为True，则使用VOC 2007的11点方法（默认：False）。
    """
    if use_07_metric:
        # VOC 2007 11点插值法
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):  # 遍历[0.0, 1.0]范围内的11个阈值
            if np.sum(rec >= t) == 0:  # 如果没有满足条件的召回率
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 取当前阈值下的最大精确率
            ap = ap + p / 11.0  # 累加AP
    else:
        # VOC 2010+ 平滑曲线法
        mrec = np.concatenate(([0.0], rec, [1.0]))  # 添加哨兵值
        mpre = np.concatenate(([0.0], prec, [0.0]))
        # 计算精度包络线
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # 找到召回率变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # 计算PR曲线下的面积
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def prec_rec_prepare(all_detected_dogs, all_gt_dogs):
    # 读取GT对象
    class_recs = {}
    npos = 0
    for imname, bbox in all_gt_dogs.items():
        difficult = np.array([False]*len(bbox))
        det = np.array([False]*len(bbox))
        n_easy = sum(~difficult) if len(difficult) else 0
        npos = npos + n_easy
        class_recs[imname] = {
                "bbox": bbox, "difficult": difficult, "det": det}

    # Read detections
    image_ids = []
    confidence = []
    BB = []
    for imname, detected_dogs in all_detected_dogs.items():
        for detected_dog in detected_dogs:
            box, score = np.array_split(detected_dog, [4])
            BB.append(box)
            confidence.append(score)
            image_ids.append(imname)
    confidence = np.hstack(confidence)
    BB = np.vstack(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    return image_ids, class_recs, BB, npos


def prec_rec_compute(image_ids, class_recs, BB, npos, ovthresh):
    # 遍历检测结果并标记TP和FP
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return prec, rec


def compute_ap_and_recall(all_detected_dogs, all_gt_dogs, ovthresh):
    """
    计算VOC检测指标。该代码改编自detectron2仓库
    """
    image_ids, class_recs, BB, npos = prec_rec_prepare(
            all_detected_dogs, all_gt_dogs)
    prec, rec = prec_rec_compute(image_ids, class_recs, BB, npos, ovthresh)
    ap = voc_ap(rec, prec, False)
    return ap, rec[-1]


def eval_stats_at_threshold(all_detected_dogs, all_gt_dogs, thresholds=[0.3, 0.4, 0.5]):
    """
    在不同的交并比（IoU）阈值下评估平均精度（AP）和召回率。
    参数：
      - all_detected_dogs: 检测到的狗框
      - all_gt_dogs: 真实标注的狗框
      - thresholds: IoU阈值列表
    返回：包含AP和召回率的DataFrame
    """
    stats = {}
    for ovthresh in thresholds:  # 遍历每个IoU阈值
        ap, recall = compute_ap_and_recall(all_detected_dogs, all_gt_dogs, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}  # 存储AP和召回率
    stats_df = pd.DataFrame.from_records(stats) * 100  # 转换为百分比形式
    return stats_df


def read_metadata(dataset):
    """
    从torch数据集中读取VOC2007元数据，避免通过dataloader循环
    """
    metadata = {}
    for anno_id, anno in enumerate(dataset.annotations):
        impath = dataset.images[anno_id]
        imname = Path(dataset.images[anno_id]).name
        xml_parsed = dataset.parse_voc_xml(
            ET.parse(anno).getroot())
        metaitem = {
                'imname': imname,
                'anno_id': anno_id,
                'impath': impath,
                'xml_parsed': xml_parsed}
        metadata[imname] = metaitem
    return metadata


def produce_gt_dog_boxes(metadata):
    """
    生成真实标注的狗框
    返回：Dict[image_name, [N_boxes, 4] np.array的框坐标]
    """
    all_gt_dogs = {}
    for imname, metaitem in metadata.items():
        objects = metaitem['xml_parsed']['annotation']['object']
        gt_dogs = []
        for obj in objects:
            if obj['name'] != 'dog':
                continue
            b = obj['bndbox']
            bbox = np.r_[int(b['xmin']), int(b['ymin']),
                int(b['xmax']), int(b['ymax'])]
            gt_dogs.append(bbox)
        gt_dogs = np.array(gt_dogs)
        all_gt_dogs[imname] = gt_dogs
    return all_gt_dogs


def produce_fake_centered_dog_boxes(metadata, scale, cheating=True):
    """
    生成带有分数=1.0的假狗框
    返回：Dict[image_name, [N_boxes, 5] np.array的框坐标+分数]
    """
    all_detected_dogs = {}
    for imname, metaitem in metadata.items():
        size = metaitem['xml_parsed']['annotation']['size']
        h, w = int(size['height']), int(size['width'])
        sq_scale = np.sqrt(scale)
        rel_box = np.r_[1-sq_scale, 1-sq_scale, 1+sq_scale, 1+sq_scale]/2
        box = rel_box * np.r_[w, h, w, h]
        detected_dogs = np.array(
                [np.r_[box, 1.0]])
        all_detected_dogs[imname] = detected_dogs
    return all_detected_dogs


def visualize_dog_boxes(folder, all_detected_dogs, all_gt_dogs, metadata):
    # 可视化预测和真实标注
    for imname, gt_dogs in all_gt_dogs.items():
        metaitem = metadata[imname]
        impath = metaitem['impath']
        img = cv2.imread(str(impath))
        detected_dogs = all_detected_dogs.get(imname, [])
        if not len(detected_dogs):
            continue
        # Draw GT dogs
        objects = metaitem['xml_parsed']['annotation']['object']
        for obj in objects:
            if obj['name'] != 'dog':
                continue
            b = obj['bndbox']
            bbox = np.r_[int(b['xmin']), int(b['ymin']),
                int(b['xmax']), int(b['ymax'])]
            visualize_bbox(
                    img, bbox, 'GT_'+obj['name'],
                    BOX_COLOR=(200, 255, 0),
                    TEXT_COLOR=(30, 30, 30))
        # Draw detected dogs
        for ind, detected_dog in enumerate(detected_dogs):
            box, score = np.array_split(detected_dog, [4])
            visualize_bbox(img, box, f'dog_{ind}_{score[0]:.2f}')
        cv2.imwrite(str(folder/imname), img)
