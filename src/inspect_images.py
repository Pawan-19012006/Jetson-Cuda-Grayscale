
import cv2
import os
import glob

# Image directory relative to the project root
DATA_DIR = "data"

image_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))

if not image_paths:
    print("No JPG images found!")
    exit()

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load: {image_path}")
        continue

    height, width, channels = image.shape

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Channels: {channels}")
    print(f"Total pixels: {width * height:,}")
    print(f"Memory: {image.nbytes / (1024 * 1024):.2f} MB")
