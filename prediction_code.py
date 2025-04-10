import cv2
import numpy as np
from ultralytics import YOLO




def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    box format: [x1, y1, x2, y2]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou

def evaluate_frame(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate a single frame by comparing predicted boxes to ground truth.
    predictions: list of dicts with keys 'bbox' and 'label'
    ground_truth: list of dicts with keys 'bbox' and 'label'
    Returns: (TP, FP, FN) counts for the frame.
    """
    TP = 0
    FP = 0
    matched_gt = set()

    for pred in predictions:
        pred_box = pred['bbox']
        pred_label = pred['label']
        best_iou = 0
        best_gt_index = -1

        for i, gt in enumerate(ground_truth):
            if i in matched_gt or gt['label'] != pred_label:
                continue
            iou = compute_iou(pred_box, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt_index)
        else:
            FP += 1

    FN = len(ground_truth) - len(matched_gt)
    return TP, FP, FN

# Load your trained YOLO model
model_path = r"C:\Users\surya\Documents\SIH traffic project\runs\detect\train5\weights\best.pt"
model = YOLO(model_path)

# Video path
video_path = r"C:\Users\surya\Documents\SIH traffic project\Test_video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Dummy ground truth dictionary for demonstration.
# The key is the frame index (integer) and the value is a list of ground truth objects.
# Each ground truth object is a dict with 'bbox' (format: [x1, y1, x2, y2]) and 'label'
ground_truth_data = {
    0: [{'bbox': [50, 60, 200, 220], 'label': 'Ambulance'},
        {'bbox': [300, 100, 400, 250], 'label': 'Car'}],
    1: [{'bbox': [55, 65, 205, 225], 'label': 'Ambulance'},
        {'bbox': [310, 110, 410, 260], 'label': 'Car'}],
    # Add entries for all frames you wish to evaluate
}

total_TP, total_FP, total_FN = 0, 0, 0
frame_index = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Optionally resize the frame
    frame = cv2.resize(frame, (700, 500))

    # Run YOLO inference
    results = model(frame)
    # Assuming results[0] holds the detections for the current frame
    boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: [N, 4]
    confidences = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Build list of predictions
    predictions = []
    for box, conf, cls in zip(boxes, confidences, classes):
        # Optionally filter out low confidence detections
        if conf < 0.5:
            continue
        # Map class index to label using the model's names attribute
        label = results[0].names[cls]
        predictions.append({'bbox': box.tolist(), 'label': label})

    # Get ground truth for the current frame if available
    if frame_index in ground_truth_data:
        gt = ground_truth_data[frame_index]
    else:
        gt = []  # or continue to next frame if no ground truth is available

    # Evaluate the frame
    TP, FP, FN = evaluate_frame(predictions, gt)
    total_TP += TP
    total_FP += FP
    total_FN += FN

    # Visualize predictions
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()



