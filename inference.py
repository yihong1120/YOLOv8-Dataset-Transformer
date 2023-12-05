from ultralytics import YOLO
import cv2

# Load a pretrained model
model = YOLO("models/best.pt")

# Predict on an image
im2 = cv2.imread("images/car.png")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels