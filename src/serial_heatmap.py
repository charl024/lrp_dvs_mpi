import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

W = 346
H = 260

packet_size = 1000

def main():
    events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")

    print(f"Loaded {len(events)} events")
    print(f"Packet Size: {packet_size}")

    heatmap = np.zeros((H, W), dtype=np.int32)

    # start time
    start_time = time.perf_counter()

    # simulate packet stream
    for i in range(0, len(events), packet_size):
        packet = events[i:i+packet_size]
        
        prc.process_heatmap(
            local_data=packet,
            heatmap=heatmap, 
            x_start=0, 
            y_start=0, 
            width=W, 
            height=H
        )

    # end time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"[TIME] Serial heatmap time: {total_time:.6f} seconds")

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Event Count")
    plt.title("Event Heatmap (Serial)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    main()
