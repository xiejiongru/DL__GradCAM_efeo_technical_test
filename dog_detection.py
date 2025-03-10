import logging

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models
from torchvision.ops import nms
import numpy as np
import cv2
from pathlib import Path

from tools.snippets import (quick_log_setup, mkdir)
from tools.voc import (VOC_ocv, transforms_voc_ocv_eval,
                       sequence_batch_collate_v2, enforce_all_seeds)
from tools.dogs import (
        eval_stats_at_threshold, read_metadata,
        produce_gt_dog_boxes, produce_fake_centered_dog_boxes,
        visualize_dog_boxes)


def dog_detection():
    """False
    Implement two simple dog detection baselines:
      A. Predict a fixed-size box at the center of each image (50% area coverage)
      B. Peek at ground truth annotations and predict a centered box only for images with dogs
    Also provide evaluation and visualization functions.
    """
    # Configuration section: Load dataset and metadata
    voc_folder = 'voc_dataset'  
    dataset_test = VOC_ocv(
        voc_folder, year='2007', image_set='test',
        download=False, transforms=transforms_voc_ocv_eval
    )
    metadata_test = read_metadata(dataset_test) 
    metadata_test = dict(list(metadata_test.items())[:500])  # Keep only the first 500 images
    all_gt_dogs = produce_gt_dog_boxes(metadata_test)  # Generate ground truth dog boxes

    # Baseline A: Predict a fixed box at the center of each image
    all_centerbox_dogs = produce_fake_centered_dog_boxes(metadata_test, scale=0.3)
    stats_df = eval_stats_at_threshold(all_centerbox_dogs, all_gt_dogs)
    log.info('A. ALL CENTERBOX dogs\n{}'.format(stats_df))

    # Baseline B: Peek at ground truth annotations and predict a centered box only for images with dogs
    all_cheating_centerbox_dogs = {
        k: v for k, v in all_centerbox_dogs.items() if len(all_gt_dogs[k])
    }
    stats_df = eval_stats_at_threshold(all_cheating_centerbox_dogs, all_gt_dogs)
    log.info('B. CHEATING CENTERBOX dogs\n{}'.format(stats_df))

    # Baseline C: Combine classifier prediction scores and assign scores to centered boxes
    initial_seed = 42  
    num_workers = 0  
    inputs_ckpt = "./model_at_epoch_019.pth.tar"  
    subset_test = torch.utils.data.Subset(dataset_test, list(range(500)))  
    dataloader_test = torch.utils.data.DataLoader(
        subset_test, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=sequence_batch_collate_v2
    )
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)  # Change output layer to 20 classes
    model.to(device)
    model.eval()
    states = torch.load(inputs_ckpt, map_location=device)
    model.load_state_dict(states["model_sdict"])
    enforce_all_seeds(initial_seed)  

    # Locate the last convolutional layer of ResNet50
    target_layer = model.layer4[-1].conv3
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Perform predictions and assign scores to centered boxes
    all_scored_centerdogs = {}
    for i_batch, (data, target, meta) in enumerate(tqdm(dataloader_test, "Evaluating test set")):
        imname = meta[0]['xml_parsed']['annotation']['filename']
        data = data.to(device)
        
        # Forward pass
        pred = torch.sigmoid(model(data))
        # dog_score = pred[0,4].item()
        # print(f"Image: {imname}, Dog score: {dog_score:.2f}")
        
        # Backward pass to compute gradients (for dog class, index 4)
        model.zero_grad()
        pred[:,4].backward(retain_graph=True)
        
        # Generate heatmap
        weights = torch.mean(gradients["value"], dim=(2,3), keepdim=True)
        cam = torch.sum(weights * activations["value"], dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=data.shape[2:], mode="bilinear")
        heatmap = cam.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Get original image dimensions
        orig_h = int(meta[0]['xml_parsed']['annotation']['size']['height'])
        orig_w = int(meta[0]['xml_parsed']['annotation']['size']['width'])
        scale_x = orig_w / data.shape[3]  
        scale_y = orig_h / data.shape[2]  

        # Thresholding and contour detection
        _, binary_map = cv2.threshold(heatmap, 0.5, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generate bounding boxes and scale coordinates
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 30:  
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Scale coordinates to original image size
            xmin = int(x * scale_x)
            ymin = int(y * scale_y)
            xmax = int((x + w) * scale_x)
            ymax = int((y + h) * scale_y)
            boxes.append([xmin, ymin, xmax, ymax, pred[0,4].item()])      
        
        all_scored_centerdogs[imname] = np.array(boxes) if boxes else np.empty((0,5))

         # Get original image path
        impath = meta[0]['impath']
        orig_img = cv2.imread(impath)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  

        # Generate heatmap color overlay
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (orig_w, orig_h))  
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_color, 0.5, 0) 

        # Draw detection boxes (red) and GT boxes (green)
        for box in boxes:
            xmin, ymin, xmax, ymax, _ = box
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        gt_boxes = all_gt_dogs.get(imname, [])
        for gt_box in gt_boxes:
            xmin, ymin, xmax, ymax = gt_box.astype(int)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor([box[:4] for box in boxes], dtype=torch.float32)
            scores = torch.tensor([box[4] for box in boxes], dtype=torch.float32)
            keep_indices = nms(boxes_tensor, scores, iou_threshold=0.5)
            boxes = [boxes[i] for i in keep_indices]

        vis_folder = mkdir('visualize/heatmap_boxes')  
        output_path = str(vis_folder / Path(impath).name)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  

    stats_df = eval_stats_at_threshold(all_scored_centerdogs, all_gt_dogs)
    log.info('C. Scored center box detection results:\n{}'.format(stats_df))
    fold = mkdir('visualize/scored_centerbox_dogs')
    visualize_dog_boxes(fold, all_scored_centerdogs, all_gt_dogs, metadata_test)

if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    dog_detection()
