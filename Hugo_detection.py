import jetson.inference
import jetson.utils
import os
import cv2

# Initialize the detection network with a threshold of 0.5
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Define the path to your images
image_folder = "/home/nvidia/jetson-inference/data/test"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_file)
    
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to RGBA format (required by jetson.utils)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Create a CUDA image from the NumPy array
    cuda_img = jetson.utils.cudaFromNumpy(img_rgba)
    
    # Perform detection
    detections = net.Detect(cuda_img)
    
    # Print detection details
    print(f"Processing image: {image_file}")
    print(f"Detected {len(detections)} objects.")
    for detection in detections:
        class_name = net.GetClassDesc(detection.ClassID)
        confidence = detection.Confidence
        left = detection.Left
        top = detection.Top
        right = detection.Right
        bottom = detection.Bottom
        width = detection.Width
        height = detection.Height
        area = detection.Area
        center = detection.Center
        print(f"ClassID: {class_name} ({detection.ClassID})")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box: Left={left:.2f}, Top={top:.2f}, Right={right:.2f}, Bottom={bottom:.2f}")
        print(f"Width: {width:.2f}, Height: {height:.2f}")
        print(f"Area: {area:.2f}")
        print(f"Center: ({center[0]:.2f}, {center[1]:.2f})")
        print("-" * 40)
    
    # Optionally, save the output image
    output_image_path = os.path.join(image_folder, f"output_{image_file}")
    output_img = jetson.utils.cudaToNumpy(cuda_img)
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(output_image_path, output_img_bgr)
    
    print(f"Processed {image_file} with {len(detections)} detections.")