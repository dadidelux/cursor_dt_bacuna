from ultralytics import YOLO

# Path to the data config file
data_yaml = 'dataset/intercropping_data.yaml'

# Choose a YOLOv8 model variant (nano for speed, small for better accuracy)
model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt for better results

# Train the model
model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    name='intercropping_yolo_model'
)

# Save the best model path for reference
print("Training complete. Best model saved in runs/detect/intercropping_yolo_model/weights/best.pt") 