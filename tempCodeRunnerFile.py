            boxes_tensor = torch.tensor([box[:4] for box in boxes])
            scores = torch.tensor([box[4] for box in boxes])
            keep_indices = nms(boxes_tensor, scores, iou_threshold=0.5)
            boxes = [boxes[i] for i in keep_indices]