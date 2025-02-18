Vehicle Detection using YOLO and OpenCV
This project implements vehicle detection in images using the YOLO (You Only Look Once) neural network, powered by OpenCV's DNN module. It detects vehicles in images and counts the total number of detected vehicles, drawing bounding boxes around each one.

Features:
Loads YOLOv4 model for vehicle detection.
Processes images to detect vehicles from predefined class IDs (car, bus, truck, etc.).
Draws bounding boxes around detected vehicles.
Counts and displays the total number of detected vehicles.
Requirements:
Python 3.x
OpenCV (opencv-python, opencv-python-headless)
NumPy

Install the required dependencies using:

"pip install opencv-python opencv-python-headless numpy"

Project Structure:
1.) vehicle_detector.py: Contains the VehicleDetector class to load and run the YOLO model for vehicle detection.
2.) dnn_model/: Folder containing the YOLOv4 weights and config files (yolov4.weights, yolov4.cfg).
3.) main.py: Main script to detect and count vehicles in images.

How to Use:
Place your images in the images2/ folder.
Run main.py to start the vehicle detection process.

"python main.py"

This will process all images in the images2/ folder, detect vehicles, draw bounding boxes around them, and display the processed images. The total count of detected vehicles will be printed at the end.

Vehicle Detection:
The detection uses the YOLOv4 model, which is capable of detecting multiple objects in an image.
Only vehicles from the following classes are detected: Car (ID 2), Motorcycle (ID 3), Bus (ID 5), Truck (ID 7).

Example Output:
For each image processed, bounding boxes are drawn around detected vehicles, and the modified image is saved with the bounding boxes visible.
Total vehicle count across all images will also be displayed.
