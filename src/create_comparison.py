import cv2
import numpy as np
import os

DATA_DIR = "data"
RESULTS_DIR = "results"

IMAGES = [
    {
        "name": "img.jpg",
        "orig_path": os.path.join(DATA_DIR, "img.jpg"),
        "cpu_path": os.path.join(RESULTS_DIR, "img_cpu_gray.jpg"),
        "cuda_path": os.path.join(RESULTS_DIR, "img_cuda_pinned_gray.jpg"),
        "display_name": "img.jpg (2268 x 4032)"
    },
    {
        "name": "img2.jpg",
        "orig_path": os.path.join(DATA_DIR, "img2.jpg"),
        "cpu_path": os.path.join(RESULTS_DIR, "img2_cpu_gray.jpg"),
        "cuda_path": os.path.join(RESULTS_DIR, "img2_cuda_pinned_gray.jpg"),
        "display_name": "img2.jpg (4000 x 2250)"
    },
    {
        "name": "img3.jpg",
        "orig_path": os.path.join(DATA_DIR, "img3.jpg"),
        "cpu_path": os.path.join(RESULTS_DIR, "img3_cpu_gray.jpg"),
        "cuda_path": os.path.join(RESULTS_DIR, "img3_cuda_pinned_gray.jpg"),
        "display_name": "img3.jpg (4211 x 4078)"
    }
]

def validate_and_compare():
    print("=" * 82)
    print("           CPU vs. CUDA PINNED GRAYSCALE NUMERICAL VALIDATION")
    print("=" * 82)
    print(f"{'Image':<10} | {'Max Diff':<9} | {'Mean Diff':<11} | {'Diff == 0 (%)':<15} | {'Diff <= 2 (%)':<15} | {'Status':<6}")
    print("-" * 82)

    rows = []
    target_w = 400
    target_h = 240

    for item in IMAGES:
        orig = cv2.imread(item["orig_path"])
        cpu = cv2.imread(item["cpu_path"], cv2.IMREAD_GRAYSCALE)
        cuda = cv2.imread(item["cuda_path"], cv2.IMREAD_GRAYSCALE)

        if orig is None or cpu is None or cuda is None:
            raise FileNotFoundError(f"Missing required file for {item['name']}")

        # Numerical diff check
        diff = cv2.absdiff(cpu, cuda)
        max_diff = int(np.max(diff))
        mean_diff = float(np.mean(diff))
        exact_pct = float(np.mean(diff == 0) * 100.0)
        le_two_pct = float(np.mean(diff <= 2) * 100.0)
        status = "PASS" if max_diff <= 2 else "FAIL"

        print(f"{item['name']:<10} | {max_diff:<9} | {mean_diff:<11.6f} | {exact_pct:<14.4f}% | {le_two_pct:<14.4f}% | {status:<6}")

        # Convert grayscale images to 3-channel for stacking with original BGR
        cpu_bgr = cv2.cvtColor(cpu, cv2.COLOR_GRAY2BGR)
        cuda_bgr = cv2.cvtColor(cuda, cv2.COLOR_GRAY2BGR)

        # Resize to fixed thumbnail dimensions for consistent grid layout
        r_orig = cv2.resize(orig, (target_w, target_h), interpolation=cv2.INTER_AREA)
        r_cpu = cv2.resize(cpu_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        r_cuda = cv2.resize(cuda_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Add annotations to thumbnails
        cv2.putText(r_orig, item["display_name"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(r_cpu, f"Max Diff: {max_diff}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(r_cuda, "CUDA Pinned", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Concatenate horizontally
        row = np.hstack([r_orig, r_cpu, r_cuda])
        rows.append(row)

    print("=" * 82)

    # Header section for columns
    header_h = 45
    header = np.full((header_h, target_w * 3, 3), 35, dtype=np.uint8)
    titles = ["Original (BGR)", "CPU Baseline (OpenCV cvtColor)", "CUDA Pinned Memory Kernel"]
    for i, title in enumerate(titles):
        pos_x = i * target_w + 20
        cv2.putText(header, title, (pos_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Build final grid
    grid = np.vstack([header] + rows)

    out_path = os.path.join(RESULTS_DIR, "visual_comparison.jpg")
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])

    file_size_kb = os.path.getsize(out_path) / 1024.0
    print(f"\n[SUCCESS] Visual comparison saved to: {out_path} ({file_size_kb:.1f} KB)")

if __name__ == "__main__":
    validate_and_compare()
