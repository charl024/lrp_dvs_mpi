import numpy as np
import processing as prc
import matplotlib.pyplot as plt

W = 346
H = 260

packet_size = 1000

def main():
    events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")

    print(f"Loaded {len(events)} events")
    print(f"Packet Size: {packet_size}")

    # time surface init
    time_surface = np.zeros((H, W), dtype=np.float64)

    # HATS block size and init
    block_w = W // 32
    block_h = H // 32
    hats = np.zeros((H // block_h, W // block_w), dtype=np.float64)

    for i in range(0, len(events), packet_size):
        packet = events[i:i+packet_size]
        # set current timestamp to the last timestamp stored in packet
        t_current = packet[-1][2]

        prc.process_hats_descriptor(
            local_data=packet,
            time_surface=time_surface,
            hats=hats,
            width=W,
            height=H,
            tau=0.02,
            local_timestamp=t_current,
            block_w=block_w,
            block_h=block_h
        )

    plt.figure(figsize=(8, 6))
    plt.title("HATS Descriptor (Block-Averaged Time Surface)")
    plt.imshow(hats, cmap='hot')
    plt.colorbar(label="HATS Value")
    plt.xlabel("HATS Block X")
    plt.ylabel("HATS Block Y")
    plt.show()

if __name__ == "__main__":
    main()
