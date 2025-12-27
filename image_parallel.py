import cv2
import numpy as np
import os
import time
import multiprocessing

# ================= Configuration Parameters =================
# The folder you just uploaded to GCP [cite: 21]
INPUT_DIR = "food-101-subset" 
OUTPUT_DIR = "output_multiprocessing"
# Tested across multiple process counts to analyze speedup [cite: 45, 73]
WORKER_COUNTS = [1, 2, 4, 8] 

def apply_image_filters(data):
    """
    This function represents the image processing pipeline[cite: 18].
    It applies 5 filters to a single image and saves the result.
    """
    image_path, output_folder = data
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    file_name = os.path.basename(image_path)
    
    # 1. Grayscale Conversion (Luminance formula) [cite: 24]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (3x3 kernel for smoothing) [cite: 25]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. Edge Detection (Sobel filter to detect edges) [cite: 26]
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
    
    # 4. Image Sharpening (Enhance edges and details) [cite: 27]
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(edges, -1, kernel)
    
    # 5. Brightness Adjustment [cite: 28]
    # Increases brightness using convertScaleAbs
    final_image = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=30)
    
    # Save the processed image to the output directory
    output_path = os.path.join(output_folder, f"multiprocess_{file_name}")
    cv2.imwrite(output_path, final_image)
    
    return file_name

def run_multiprocessing_test(image_paths, workers):
    """
    Manages the process pool and records execution time[cite: 32, 39].
    """
    # Preparing data pairs for the map function
    tasks = [(path, OUTPUT_DIR) for path in image_paths]
    
    start_time = time.time()
    
    # 'multiprocessing.Pool' handles the Data Parallelism [cite: 30, 39]
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(apply_image_filters, tasks)
        
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get all valid images from the subset folder
    if os.path.exists(INPUT_DIR):
        all_images = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not all_images:
            print(f"❌ No images found in {INPUT_DIR}")
        else:
            print(f"🚀 Starting Performance Analysis on {len(all_images)} images...")
            results = {}

            # Execute the pipeline with different numbers of workers 
            for count in WORKER_COUNTS:
                print(f"🔄 Processing with {count} worker(s)...")
                exec_time = run_multiprocessing_test(all_images, count)
                results[count] = exec_time

            # --- PERFORMANCE TABLE GENERATION [cite: 45, 73] ---
            print("\n" + "="*65)
            print(f"{'Workers':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<10}")
            print("-" * 65)
            
            t_serial = results[1] # Time with 1 worker is the baseline
            
            for p in WORKER_COUNTS:
                t_p = results[p]
                speedup = t_serial / t_p # Formula: S = T1 / Tp
                efficiency = (speedup / p) * 100 # Formula: E = (S / p) * 100
                print(f"{p:<10} | {t_p:<12.4f} | {speedup:<10.2f} | {efficiency:<10.2f}%")
            
            print("="*65)
            print("✅ Performance data collected. Ready for technical report.")
    else:
        print(f"❌ Input directory '{INPUT_DIR}' not found. Please upload it to GCP.")
