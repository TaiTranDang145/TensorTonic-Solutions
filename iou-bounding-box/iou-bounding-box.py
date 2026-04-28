def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    if box_a[2] < box_b[0]:
        return 0
    intersection = (min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * (min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    union = (box_a[2] - box_a[0])*(box_a[3] - box_a[1]) + (box_b[2] -box_b[0]) *(box_b[3] - box_b[1]) - intersection
    return intersection/union