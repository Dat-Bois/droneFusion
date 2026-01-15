import os
import glob
import pickle
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_sort_files(folder_path):
    file_pattern = os.path.join(folder_path, "radar_*.pkl")
    files = glob.glob(file_pattern)
    data_files = []
    print(f"Found {len(files)} files in {folder_path}...")
    
    for fpath in files:
        filename = os.path.basename(fpath)
        match = re.search(r'radar_(\d+)\.pkl', filename)
        if match:
            timestamp_us = int(match.group(1))
            data_files.append((timestamp_us, fpath))
    
    data_files.sort(key=lambda x: x[0])
    return data_files

def visualize_radar_data(folder_path):
    sorted_files = load_and_sort_files(folder_path)
    if not sorted_files:
        print("No matching files found.")
        return

    plt.ion()  
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    MAX_RANGE_METERS = 10.0 
    ax.set_xlim(-MAX_RANGE_METERS, MAX_RANGE_METERS)
    ax.set_ylim(0, MAX_RANGE_METERS) # assuming forward is +Y TODO: verify
    ax.set_zlim(-2, 2)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title("Radar Point Cloud Playback")

    scat = ax.scatter([], [], [], c=[], cmap='jet', s=10)

    print("Starting playback...")
    start_wall_time = time.time()
    first_file_ts = sorted_files[0][0] 

    for i, (current_ts, fpath) in enumerate(sorted_files):
        recording_delta_sec = (current_ts - first_file_ts) / 1e6
        wall_delta_sec = time.time() - start_wall_time
        sleep_duration = recording_delta_sec - wall_delta_sec
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
        point_cloud = data.get('pointCloud')
        if point_cloud is None or point_cloud.shape[0] == 0:
            continue

        # [x, y, z, doppler, snr, noise, track_id]
        xs = point_cloud[:, 0]
        ys = point_cloud[:, 1]
        zs = point_cloud[:, 2]
        doppler = point_cloud[:, 3] 

        scat._offsets3d = (xs, ys, zs)
        # normalize doppler -2m/s to 2m/s
        scat.set_array(doppler) 
        scat.set_clim(-2, 2) 
        ax.set_title(f"Frame: {i} | Time: {recording_delta_sec:.2f}s | Points: {len(xs)}")
        fig.canvas.draw_idle()
        plt.pause(0.001)
    print("Playback finished.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    DATA_FOLDER = "/home/eesh/droneFusion/dev/RECORD/01_14/test_01_14_17_32"  
    if not os.path.exists(DATA_FOLDER):
        print(f"Folder '{DATA_FOLDER}' not found. Please update DATA_FOLDER path.")
    else:
        visualize_radar_data(DATA_FOLDER)