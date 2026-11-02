
import cv2 #Opencv: Loads, processes and saves image
import os #Handle folder and file paths
import glob #Find file matching a pattern
import time #Measure execution time

DATA_DIR = "data" #where our original images are stored.
RESULTS_DIR = "results" #where our processed images will be saved.

os.makedirs(RESULTS_DIR, exist_ok=True) #This creates the results directory if it doesn't already exist.

image_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg")) #Finds all the jpg images, returns their paths as list

for image_path in image_paths:
    image = cv2.imread(image_path) #Loops through the list, loads the image and decodes it into pixel data,The result is an image represented as a NumPy array.

    if image is None:
        print(f"Could not load {image_path}")
        continue

    # Warm-up
    for _ in range(5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #The first execution of an operation may behave differently from subsequent executions because of things like initial setup, memory allocation, and CPU cache state.Warm-up helps reduce the influence of initial execution effects on our measurements.

    # Benchmark
    iterations = 100
    start = time.perf_counter() #time.perf_counter() is a Python timer designed for measuring short time intervals with high precision.

    for _ in range(iterations):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #cv2.cvtColor() is an OpenCV function that converts an image from one colour representation to another.

    end = time.perf_counter()

    avg_time_ms = ((end - start) / iterations) * 1000 #Calculating average execution time
    fps = 1000 / avg_time_ms #It tells us how many images could theoretically be processed per second if each conversion takes the measured average time.

    filename = os.path.basename(image_path) #Extracts img2.jpg (example)
    output_path = os.path.join(
        RESULTS_DIR,
        os.path.splitext(filename)[0] + "_cpu_gray.jpg" #separates the filename and extension
    )

    cv2.imwrite(output_path, gray)

    print(f"\nImage: {filename}")
    print(f"Average CPU time: {avg_time_ms:.4f} ms")
    print(f"CPU FPS: {fps:.2f}")
    print(f"Output saved: {output_path}")
