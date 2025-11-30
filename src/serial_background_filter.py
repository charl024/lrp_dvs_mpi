import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

# convert events into a heatmap for visualization
def transform_to_heatmap(filtered_data, width, height):
    filtered_heatmap = np.zeros((height, width), dtype=np.int32)

    xs = filtered_data[:, 0].astype(int)
    ys = filtered_data[:, 1].astype(int)

    mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[mask]
    ys = ys[mask]

    for x, y in zip(xs, ys):
        filtered_heatmap[y, x] += 1

    return filtered_heatmap


def serial_background_filter(events, width, height, packet_size=1000, T_thresh=2000.0, display=False):

    # grid of local timestamps
    local_timestamp = np.full((height, width), -1.0, dtype=np.float64)

    # start time
    start_time = time.perf_counter()

    # to store filtered events
    filtered = []

    for i in range(0, len(events), packet_size):
        packet = events[i:i+packet_size]
        out = prc.process_background_filter(
            local_data=packet,
            local_timestamp=local_timestamp,
            x_start=0,
            y_start=0,
            width=width,
            height=height,
            T_thresh=T_thresh
        )
        
        filtered.append(out)
    
    # combine all arrays of filtered events
    filtered = np.concatenate(filtered, axis=0)

    # end time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"[TIME] Serial Background Filtering time: {total_time:.6f} seconds")

    if not display:
        return

    # convert to heatmap for display
    filtered_heatmap = transform_to_heatmap(filtered_data=filtered, width=width, height=height)

    plt.figure(figsize=(8, 6))
    filtered_heatmap_log = np.log1p(filtered_heatmap)

    # visualization
    plt.imshow(
        filtered_heatmap_log,
        cmap="hot",
        extent=[0, width, height, 0]
    )

    plt.colorbar(label="Event Density")
    plt.title("Filtered Event Heatmap")
    plt.xlabel("X coordinate (pixels)")
    plt.ylabel("Y coordinate (pixels)")
    plt.tight_layout()
    plt.savefig("plots/serial/background_filter_out.svg")