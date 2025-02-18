# Import necessary libraries
import cv2  # OpenCV library for computer vision
import numpy as np  # NumPy library for numerical operations

# Define a class for vehicle detection
class VehicleDetector:

    def __init__(self):
        # Load YOLO (You Only Look Once) Neural Network
        net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
        
        # Create an instance of cv2.dnn_DetectionModel and set the network
        self.model = cv2.dnn_DetectionModel(net)
        
        # Set input parameters for the model
        self.model.setInputParams(size=(832, 832), scale=1 / 255)

        # Define a list of class IDs representing vehicles
        self.classes_allowed = [2, 3, 5, 6, 7]

    def detect_vehicles(self, img):
        # Detect objects in the input image using the YOLO model
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        
        # Iterate through the detected objects
        for class_id, score, box in zip(class_ids, scores, boxes):
            # Skip detection with low confidence
            if score < 0.5:
                continue

            # Include only the detected vehicles (classes specified in self.classes_allowed)
            if class_id in self.classes_allowed:
                vehicles_boxes.append(box)

        # Return the list of bounding boxes around detected vehicles
        return vehicles_boxes
