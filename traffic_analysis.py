import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText

from PIL import Image

# -----------------------------------------------------------
# Part 1: YOLOv8 Vehicle Detection and Evaluation Code
# -----------------------------------------------------------
def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Box format: [x1, y1, x2, y2]
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

# Load your trained YOLO model (adjust the path as necessary)
model_path = r"C:\Users\surya\Documents\SIH traffic project\runs\detect\train5\weights\best.pt"
yolo_model = YOLO(model_path)

# Video path and capture
video_path = r"C:\Users\surya\Documents\SIH traffic project\Test_video_1.mp4"
cap = cv2.VideoCapture(video_path)

# Dummy ground truth dictionary for demonstration
ground_truth_data = {
    0: [{'bbox': [50, 60, 200, 220], 'label': 'Ambulance'},
        {'bbox': [300, 100, 400, 250], 'label': 'Car'}],
    1: [{'bbox': [55, 65, 205, 225], 'label': 'Ambulance'},
        {'bbox': [310, 110, 410, 260], 'label': 'Car'}],
    # Add additional frame entries as needed.
}

total_TP, total_FP, total_FN = 0, 0, 0
frame_index = 0

# We will also capture one example annotated frame for use with the vision model.
example_annotated_frame = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Optionally resize the frame
    frame = cv2.resize(frame, (700, 500))

    # Run YOLO inference
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2]
    confidences = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    # Build list of predictions for the current frame
    predictions = []
    for box, conf, cls in zip(boxes, confidences, classes):
        if conf < 0.5:
            continue  # Filter low confidence detections
        label = results[0].names[cls]
        predictions.append({'bbox': box.tolist(), 'label': label})

    # Get ground truth for the frame if available
    gt = ground_truth_data.get(frame_index, [])

    # Evaluate the frame
    TP, FP, FN = evaluate_frame(predictions, gt)
    total_TP += TP
    total_FP += FP
    total_FN += FN

    # Visualize predictions (optional)
    annotated_frame = results[0].plot()

    # Save one annotated frame for vision-based report generation (for instance, the first frame)
    if example_annotated_frame is None:
        example_annotated_frame = annotated_frame.copy()

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------
# Part 2: Generate Report Using Hugging Face Transformers Pipeline
# -----------------------------------------------------------
# Login via huggingface-cli (run in terminal) before executing this code if required.
# We now use the Llama-3.2-11B-Vision model for multimodal (vision and text) generation.

# Create the pipeline for image-text-to-text tasks using the Llama-3.2-11B-Vision model.
# Optionally, include use_auth_token=True if your model access requires it.
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision", use_auth_token=True)
vision_model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision", use_auth_token=True)
vision_pipeline = pipeline("image-text-to-text", model=vision_model, processor=processor)

# Build the report prompt using the aggregated metrics
report_prompt = (
    f"Vehicle Detection Evaluation Report:\n"
    f"Total True Positives: {total_TP}\n"
    f"Total False Positives: {total_FP}\n"
    f"Total False Negatives: {total_FN}\n\n"
    "Based on these metrics, provide a detailed analysis of the detection performance, "
    "suggest improvements, and discuss any potential next steps for further refinement."
)

# Convert the example annotated frame (BGR from OpenCV) to a PIL image (RGB)
# This image will be used as the visual context for the vision model.
if example_annotated_frame is None:
    raise ValueError("No annotated frame is available for generating the vision-based report.")
pil_image = Image.fromarray(cv2.cvtColor(example_annotated_frame, cv2.COLOR_BGR2RGB))

# Generate the report
# The pipeline takes both an image and a text prompt as input.
generated = vision_pipeline(image=pil_image, text=report_prompt,
                            max_length=300, do_sample=True, temperature=0.7)

# The output is typically a list of dictionaries; extract the text from the first item.
report_text = generated[0].get("generated_text", "")
print("\n--- Generated Report ---\n")
print(report_text)
