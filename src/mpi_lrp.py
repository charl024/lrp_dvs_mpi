from mpi4py import MPI
import numpy as np

W = 640
H = 480

Pw = 2
Ph = 2

region_width  = W // Pw
region_height = H // Ph

packet_size = 1000

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if Pw * Ph != size:
        if rank == 0:
            print("Error: Num Processes must equal ", Pw * Ph)
        return

    rx = rank % Pw
    ry = rank // Pw

    x_start = rx * region_width
    x_end = (rx + 1) * region_width if rx != Pw - 1 else W

    y_start = ry * region_height
    y_end = (ry + 1) * region_height if ry != Ph - 1 else H

    print(f"Rank {rank} owns region x[{x_start}:{x_end}], y[{y_start}:{y_end}]")

    if rank == 0:
        events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")

        print(f"Rank 0: loaded {len(events)} events")
        print(f"Rank 0: distributing events in packets of {packet_size}")

        local_count = 0

        for i in range(0, len(events), packet_size):
            packet = events[i:i+packet_size]

            buckets = {r: [] for r in range(size)}

            for x, y, t, p in packet:
                tx = min(x // region_width,  Pw - 1)
                ty = min(y // region_height, Ph - 1)
                r  = ty * Pw + tx
                buckets[r].append((x, y, t, p))

            local_data_r0 = buckets[0]
            local_count += len(local_data_r0)

            for r in range(1, size):
                bucket_arr = np.array(buckets[r], dtype=packet.dtype)
                comm.send(bucket_arr, dest=r, tag=0)
                # print(f"Rank 0: sent {len(buckets[r])} events to rank {r}")

        for r in range(1, size):
            comm.send(None, dest=r, tag=1)

        # print(f"Rank 0: processed {local_count} events locally")

    else:
        while True:
            status = MPI.Status()
            local_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            current_tag = status.Get_tag()

            if current_tag == 1:
                print(f"Rank {rank}: received end of data stream signal")
                break

            # print(f"Rank {rank}: received {len(local_data)} events")
            # process one packet

    comm.Barrier()
    if rank == 0:
        print("Finished, all is done.")

if __name__ == "__main__":
    main()
