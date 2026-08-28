from ultralytics import YOLO

# Load the small YOLO model
model = YOLO("yolo11n.pt")

# Use webcam
model.predict(
    source=0,
    show=True
)