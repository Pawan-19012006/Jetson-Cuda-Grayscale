
import cv2
import os
import glob
import time

DATA_DIR = "data"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

image_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load {image_path}")
        continue

    # Warm-up
    for _ in range(5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Benchmark
    iterations = 100
    start = time.perf_counter()

    for _ in range(iterations):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    end = time.perf_counter()

    avg_time_ms = ((end - start) / iterations) * 1000
    fps = 1000 / avg_time_ms

    filename = os.path.basename(image_path)
    output_path = os.path.join(
        RESULTS_DIR,
        os.path.splitext(filename)[0] + "_cpu_gray.jpg"
    )

    cv2.imwrite(output_path, gray)

    print(f"\nImage: {filename}")
    print(f"Average CPU time: {avg_time_ms:.4f} ms")
    print(f"CPU FPS: {fps:.2f}")
    print(f"Output saved: {output_path}")
