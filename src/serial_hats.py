import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

def serial_hats(events, width, height, packet_size=1000, tau=0.02, rho=2, block_size=32, display=False, id=1):

    # number of cells
    n_cells_y = (height + block_size - 1) // block_size
    n_cells_x = (width + block_size - 1) // block_size

    # init hats
    hats_total = np.zeros((n_cells_y, n_cells_x, 2), dtype=np.float64)

    # start time
    start = time.perf_counter()

    # simulate packet stream
    for i in range(0, len(events), packet_size):
        packet = events[i:i+packet_size]

        hats_patch = prc.process_hats_descriptor(
            events=packet,
            width=width,
            height=height,
            block_size=block_size,
            rho=rho,
            tau=tau
        )

        hats_total += hats_patch

    # end time
    end = time.perf_counter()
    total = end - start

    print(f"[TIME] Serial HATS descriptor time: {total:.6f} seconds")

    if not display:
        return

    # visualization
    hats_pos = hats_total[:, :, 1]
    hats_neg = hats_total[:, :, 0]
    hats_sum = hats_pos + hats_neg

    plt.figure(figsize=(8, 6))

    plt.imshow(
        hats_sum,
        cmap="hot",
        extent=[0, n_cells_x, n_cells_y, 0]
    )

    plt.colorbar(label="HATS Intensity")
    plt.title("HATS Descriptor (Block-Averaged Time Surface)")
    plt.xlabel("Block X Index")
    plt.ylabel("Block Y Index")
    plt.tight_layout()
    plt.savefig(f"plots/serial/hats_out{id}.svg")
    plt.close()