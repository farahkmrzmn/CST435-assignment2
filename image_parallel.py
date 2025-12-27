import cv2
import numpy as np
import os
import time
import multiprocessing

# ================= Configuration Parameters =================
# INPUT_DIR: The source folder containing your 250 Food-101 images.
INPUT_DIR = "food-101-subset" 

# OUTPUT_DIR: Where the filtered images will be saved after processing.
OUTPUT_DIR = "output_multiprocessing"

# WORKER_COUNTS: The number of CPU cores we will test (1, 2, 4, 8) to measure performance.
# Testing '1' is essential to get a baseline for 'Speedup' calculations.
WORKER_COUNTS = [1, 2, 4, 8] 

def apply_image_filters(data):
    """
    Core function that runs inside each worker process. 
    It takes an image path, applies 5 filters, and saves the result.
    """
    image_path, output_folder = data
    
    # Load the image from the disk using OpenCV
    img = cv2.imread(image_path)
    
    # If the image fails to load, skip it to prevent the program from crashing
    if img is None:
        return None
    
    # Get the original filename (e.g., 'sushi_123.jpg')
    file_name = os.path.basename(image_path)

    # --- [FILTER 1] Grayscale Conversion ---
    # Converts a color (BGR) image to a black-and-white (Grayscale) image.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- [FILTER 2] Gaussian Blur ---
    # Smoothes the image using a 3x3 kernel. This reduces 'noise' and detail.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # --- [FILTER 3] Edge Detection (Sobel) ---
    # Calculates the 'gradient' or intensity change in X and Y directions.
    # It highlights the outlines (edges) of the food items.
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
    
    # --- [FILTER 4] Image Sharpening ---
    # Uses a custom matrix (kernel) to make the edges look sharper and more distinct.
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(edges, -1, sharpen_kernel)
    
    # --- [FILTER 5] Brightness Adjustment ---
    # Increases the pixel values by a constant (beta=30) to make the image brighter.
    final_image = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=30)
    
    # Save the processed image with a unique prefix
    output_path = os.path.join(output_folder, f"multiprocess_{file_name}")
    cv2.imwrite(output_path, final_image)
    
    return file_name

def run_multiprocessing_test(image_paths, workers):
    """
    Creates a pool of 'Worker' processes and distributes the images among them.
    This is known as Data Parallelism.
    """
    # Create pairs of (image_path, output_folder) for the multiprocessing 'map' function
    tasks = [(path, OUTPUT_DIR) for path in image_paths]
    
    start_time = time.time()
    
    # 'multiprocessing.Pool' creates several copies of this Python program to run at once.
    # 'pool.map' automatically assigns different images to different workers.
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(apply_image_filters, tasks)

    # Calculate how many seconds it took to process all images
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    # Create the output directory on the GCP VM if it doesn't already exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Gather all images (.jpg, .png, .jpeg) from the input folder
    if os.path.exists(INPUT_DIR):
        all_images = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not all_images:
            print(f"❌ No images found in {INPUT_DIR}")
        else:
            print(f"🚀 Starting Performance Analysis on {len(all_images)} images...")
            results = {}

            # Test our performance using 1, 2, 4, and 8 processes
            for count in WORKER_COUNTS:
                print(f"🔄 Processing with {count} worker(s)...")
                exec_time = run_multiprocessing_test(all_images, count)
                results[count] = exec_time

            # --- PERFORMANCE TABLE GENERATION ---
            # This table prints the final metrics required for the rubric.
            print("\n" + "="*65)
            print(f"{'Workers':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15}")
            print("-" * 65)
            
            # T_SERIAL: The time taken by 1 worker (the baseline).
            t_serial = results[1] 

            for p in WORKER_COUNTS:
                t_p = results[p]
                
                # SPEEDUP: How many times faster is 'p' workers compared to 1?
                speedup = t_serial / t_p 
                
                # EFFICIENCY: Percentage of the CPU power being effectively used.
                efficiency = (speedup / p) * 100 
                
                print(f"{p:<10} | {t_p:<12.4f} | {speedup:<10.2f} | {efficiency:<15.2f}%")

            print("="*65)
            print("✅ Performance data collected. Ready for technical report.")
    else:
        print(f"❌ Input directory '{INPUT_DIR}' not found. Please upload it to GCP.")
