from ultralytics import YOLO

model = YOLO("yolov8x")

model.predict("imput_videos/image.png", save=True)
