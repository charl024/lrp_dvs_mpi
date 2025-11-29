from serial_heatmap import serial_heatmap
from serial_background_filter import serial_background_filter

from util import get_dataset

def main():
    packet_size = 1000

    width_large, height_large = 1280, 720
    width_small, height_small = 346, 260

    small_cam_dataset_path = "test_data/A62P20C3-2021_11_06_18_33_41.npy"
    large_cam_dataset_path = "test_data/Normal_Videos_003_x264.npz"

    small_cam_events = get_dataset(small_cam_dataset_path)
    large_cam_events = get_dataset(large_cam_dataset_path)

    # serial_heatmap(small_cam_events, width_small, height_small, packet_size, display=True)
    # serial_heatmap(large_cam_events, width_large, height_large, packet_size, display=True)

    serial_background_filter(small_cam_events, width_small, height_small, packet_size, T_thresh=1000.0, display=True)
    serial_background_filter(large_cam_events, width_large, height_large, packet_size, T_thresh=1000.0, display=True)


if __name__ == "__main__":
    main()
