from ultralytics import YOLO
import sys
import os

# Path to the trained model (update if needed)
MODEL_PATH = 'runs/detect/intercropping_yolo_model/weights/best.pt'

# Load the model
model = YOLO(MODEL_PATH)

# Function to run inference and summarize detections
def infer_and_summarize(image_path):
    results = model(image_path)
    summary = {}
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        summary[class_name] = summary.get(class_name, 0) + 1
    print(f"\nSummary for {os.path.basename(image_path)}:")
    if summary:
        for plant, count in summary.items():
            print(f"  {plant}: {count}")
    else:
        print("  No plants detected.")
    return summary

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python infer_and_summarize_intercropping.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    infer_and_summarize(image_path) 