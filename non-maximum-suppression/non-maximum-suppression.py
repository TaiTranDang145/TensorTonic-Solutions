def iou(box1, box2):
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # Areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0


def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    Returns indices of selected boxes.
    """
    # Sort indices by score (descending)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []

    while indices:
        # Pick highest score
        current = indices.pop(0)
        keep.append(current)

        # Filter remaining boxes
        remaining = []
        for i in indices:
            if iou(boxes[current], boxes[i]) < iou_threshold:
                remaining.append(i)

        indices = remaining

    return keep