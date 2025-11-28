import numpy as np
import processing as prc
import matplotlib.pyplot as plt

W = 346
H = 260

packet_size = 1000
T_thresh = 1e-2

# convert events into a heatmap for visualization
def transform_to_heatmap(filtered_data, width, height):
    filtered_heatmap = np.zeros((height, width), dtype=np.int32)

    xs = filtered_data[:, 0].astype(int)
    ys = filtered_data[:, 1].astype(int)

    # clip out-of-bounds (optional but safer)
    mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[mask]
    ys = ys[mask]

    for x, y in zip(xs, ys):
        filtered_heatmap[y, x] += 1

    return filtered_heatmap


def main():
    events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")

    print(f"Loaded {len(events)} events")
    print(f"Packet Size: {packet_size}")

    local_timestamp = np.full((H, W), -1.0, dtype=np.float64)

    filtered = []

    for i in range(0, len(events), packet_size):
        packet = events[i:i+packet_size]
        out = prc.process_background_filter(
            local_data=packet,
            local_timestamp=local_timestamp,
            x_start=0,
            y_start=0,
            width=W,
            height=H,
            T_thresh=T_thresh
        )
        filtered.append(out)
    
    filtered = np.concatenate(filtered, axis=0)

    # convert to heatmap for display
    filtered_heatmap = transform_to_heatmap(filtered_data=filtered, width=W, height=H)
    plt.figure(figsize=(8, 6))
    plt.imshow(filtered_heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Event Count")
    plt.title("Filtered Event Heatmap (Serial)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()    
        


if __name__ == "__main__":
    main()
