import numpy as np

W = 640
H = 480

packet_size = 1000

def main():
    events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")

    print(f"Loaded {len(events)} events")
    print(f"Packet Size: {packet_size}")

    for i in range(0, len(events), packet_size):
        packet = events[i:i+packet_size]
        print(f"Packet of size {len(packet)}")

        # process one packet



if __name__ == "__main__":
    main()
