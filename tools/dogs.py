import logging
import xml.etree.ElementTree as ET

import cv2  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from pathlib import Path
from .snippets import (visualize_bbox)

log = logging.getLogger(__name__)


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # VOC 2007 11-point interpolation method
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):  # Iterate over 11 threshold values from 0.0 to 1.0
            if np.sum(rec >= t) == 0:  # If no recall values meet the threshold
                p = 0
            else:
                p = np.max(prec[rec >= t])  # Take the maximum precision at or above the threshold
            ap = ap + p / 11.0  # Accumulate AP
    else:
        # VOC 2010+ area under the curve method
        mrec = np.concatenate(([0.0], rec, [1.0]))  # Add sentinel values
        mpre = np.concatenate(([0.0], prec, [0.0]))
        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # Identify recall level changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # Compute the area under the precision-recall curve
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def prec_rec_prepare(all_detected_dogs, all_gt_dogs):
    # Read GT objects
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
    # Iterate over detections and mark TPs and FPs
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
    Compute VOC detection metrics. This code is adapted from the detectron2 repository
    """
    image_ids, class_recs, BB, npos = prec_rec_prepare(
            all_detected_dogs, all_gt_dogs)
    prec, rec = prec_rec_compute(image_ids, class_recs, BB, npos, ovthresh)
    ap = voc_ap(rec, prec, False)
    return ap, rec[-1]


def eval_stats_at_threshold(all_detected_dogs, all_gt_dogs, thresholds=[0.3, 0.4, 0.5]):
    """
    Evaluate average precision (AP) and recall at different intersection-over-union (IoU) thresholds.
    Parameters:
      - all_detected_dogs: Detected dog bounding boxes
      - all_gt_dogs: Ground truth dog bounding boxes
      - thresholds: List of IoU thresholds
    Returns: DataFrame containing AP and recall
    """
    stats = {}
    for ovthresh in thresholds:  # Iterate over each IoU threshold
        ap, recall = compute_ap_and_recall(all_detected_dogs, all_gt_dogs, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}  # Store AP and recall
    stats_df = pd.DataFrame.from_records(stats) * 100  # Convert to percentage form
    return stats_df


def read_metadata(dataset):
    """
    Read VOC2007 metadata from a torch dataset to avoid looping through dataloader
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
    Generate ground truth dog bounding boxes
    Returns: Dict[image_name, [N_boxes, 4] np.array of bounding box coordinates]
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
    Generate fake dog bounding boxes with score=1.0
    Returns: Dict[image_name, [N_boxes, 5] np.array of bounding box coordinates + score]
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
    # Visualize predictions and ground truth
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