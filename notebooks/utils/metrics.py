
# IoU-Berechnung
def calculate_iou(pred, target, cls):
    pred_mask = (pred == cls)
    target_mask = (target == cls)
    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()
    if union > 0:
        return intersection / union
    return 0
