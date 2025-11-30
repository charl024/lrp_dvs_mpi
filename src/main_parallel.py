from parallel_heatmap import parallel_heatmap
from parallel_background_filter import parallel_background_filter
from parallel_hats import parallel_hats
import argparse
from util import get_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--process-width", type=int, required=True)
    p.add_argument("--process-height", type=int, required=True)
    p.add_argument("--packet-size", type=int, default=1000)
    return p.parse_args()

def main():

    args = parse_args()

    packet_size = args.packet_size
    process_width = args.process_width
    process_height = args.process_height

    width_large, height_large = 1280, 720
    width_small, height_small = 346, 260

    display = False

    small_cam_dataset_path = "test_data/A62P20C3-2021_11_06_18_33_41.npy"
    large_cam_dataset_path = "test_data/Normal_Videos_003_x264.npz"

    small_cam_events = get_dataset(small_cam_dataset_path)
    large_cam_events = get_dataset(large_cam_dataset_path)

    parallel_heatmap(
        small_cam_events, 
        width_small, 
        height_small,
        process_width,
        process_height,
        packet_size, 
        display=display
    )

    parallel_heatmap(
        large_cam_events, 
        width_large, 
        height_large,
        process_width,
        process_height,
        packet_size, 
        display=display
    )

    parallel_background_filter(
        small_cam_events, 
        width_small, 
        height_small,
        process_width,
        process_height,
        packet_size, 
        T_thresh=1000.0, 
        display=display
    )

    parallel_background_filter(
        large_cam_events, 
        width_large, 
        height_large, 
        process_width,
        process_height,
        packet_size, 
        T_thresh=1000.0, 
        display=display
    )

    parallel_hats(
        small_cam_events,
        width_small,
        height_small,
        process_width,
        process_height,
        block_size=8,
        packet_size=packet_size,
        tau=0.02,
        display=display
    )

    parallel_hats(
        large_cam_events,
        width_large,
        height_large,
        process_width,
        process_height,
        block_size=8,
        packet_size=packet_size,
        tau=0.02,
        rho=2,
        display=display
    )


if __name__ == "__main__":
    main()
