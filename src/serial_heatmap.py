import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

def serial_heatmap(events, width, height, packet_size=1000, display=False):

    # heatmap
    heatmap = np.zeros((height, width), dtype=np.int32)

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
            width=width, 
            height=height
        )

    # end time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"[TIME] Serial Heatmap time: {total_time:.6f} seconds")

    if not display:
        return

    plt.figure(figsize=(8, 6))
    heatmap_log = np.log1p(heatmap)
    plt.imshow(heatmap_log, cmap="hot")
    plt.colorbar()
    plt.title("Event Heatmap (Serial)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()