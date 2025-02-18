# Import necessary libraries
import cv2  # OpenCV library for computer vision
import glob  # Module for finding pathnames matching a specified pattern
from vehicle_detector import VehicleDetector  # Custom class for vehicle detection

# Load Vehicle Detector
vd = VehicleDetector()  # Create an instance of the VehicleDetector class

# Load images from a folder (assumes images are in a folder named 'images' and have the '.jpg' extension)
images_folder = glob.glob("images2/4.jpg")

# Initialize a counter for the total number of detected vehicles across all images
vehicles_folder_count = 0

# Loop through each image in the specified folder
for img_path in images_folder:
    # Print the current image path
   
   
   
    # Read the image using OpenCV
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    
    print("Img path", img_path)
    print("Image Dimensions:")
    print("Height:", height)
    print("Width:", width)
    print("Number of Channels:", channels)

    # Detect vehicles in the image using the VehicleDetector class
    vehicle_boxes = vd.detect_vehicles(img)
    # Get the count of detected vehicles in the current image
    vehicle_count = len(vehicle_boxes)
    
    # Update the total count of detected vehicles across all images
    vehicles_folder_count += vehicle_count

    # Draw bounding boxes and labels for each detected vehicle
    for box in vehicle_boxes:
        x, y, w, h = box

        # Draw a rectangle around the detected vehicle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 87, 51), 3)

        # Add a text label indicating the total count of vehicles in the image
      #  cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    # Display the image with bounding boxes and labels
    cv2.imshow("Cars", img)
    # Save the modified image with bounding boxes
    cv2.imwrite('1.jpg', img)
    # Wait for a key press before moving on to the next image
    cv2.waitKey(0)

# Print the total count of detected vehicles across all images
print("Total current count", vehicles_folder_count)
