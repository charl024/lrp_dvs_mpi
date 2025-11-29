from mpi4py import MPI
import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

W = 346
H = 260

Pw = 8
Ph = 8

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

    # print(f"Rank {rank} owns region x[{x_start}:{x_end}], y[{y_start}:{y_end}]")

    if rank == 0:
        events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")
        print(f"Rank 0: loaded {len(events)} events")
        print(f"Rank 0: distributing events in packets of {packet_size}")

    comm.Barrier()
    start_time = MPI.Wtime()

    if rank == 0:

        # simulate packet stream
        for i in range(0, len(events), packet_size):
            packet = events[i:i+packet_size]

            buckets = {r: [] for r in range(size)}

            for x, y, t, p in packet:
                tx = min(x // region_width,  Pw - 1)
                ty = min(y // region_height, Ph - 1)
                r  = ty * Pw + tx
                buckets[r].append((x, y, t, p))

            # send buckets to other thanks
            for r in range(1, size):
                bucket_arr = np.array(buckets[r], dtype=packet.dtype)
                comm.send(bucket_arr, dest=r, tag=0)
                # print(f"Rank 0: sent {len(buckets[r])} events to rank {r}")

            # init rank 0's local heatmap
            local_heatmap = np.zeros((region_height, region_width), dtype=np.int32)

            # processing on rank 0
            local_data_r0 = buckets[0]

            prc.process_heatmap(
                local_data=local_data_r0,
                heatmap=local_heatmap,
                x_start=x_start, 
                y_start=y_start, 
                width=region_width, 
                height=region_height
            )

            

        for r in range(1, size):
            comm.send(None, dest=r, tag=1)

        # print(f"Rank 0: processed {local_count} events locally")

    else:

        # init local heatmap
        local_heatmap = np.zeros((region_height, region_width), dtype=np.int32)

        while True:
            status = MPI.Status()
            local_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            current_tag = status.Get_tag()

            if current_tag == 1:
                # print(f"Rank {rank}: received end of data stream signal")
                break

            # process one packet
            prc.process_heatmap(
                local_data=local_data,
                heatmap=local_heatmap,
                x_start=x_start, 
                y_start=y_start, 
                width=region_width, 
                height=region_height
            )

    # gather heatmaps at rank 0
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((size, region_height, region_width), dtype=np.int32)

    comm.Gather(local_heatmap, recvbuf, root=0)

    comm.Barrier()
    end_time = MPI.Wtime()

    # plt.imshow(local_heatmap, cmap="hot")
    # plt.title(f"Rank {rank} Heatmap")
    # plt.colorbar()
    # plt.show()

    # calculate total time taken to process, create full_heatmap and then display
    if rank == 0:
        total_time = end_time - start_time
        print(f"[TIME] Total MPI pipeline time: {total_time:.6f} seconds")

        full_heatmap = np.zeros((H, W), dtype=np.int32)

        for r in range(size):
            rx = r % Pw
            ry = r // Pw

            x0 = rx * region_width
            y0 = ry * region_height

            full_heatmap[y0:y0+region_height, x0:x0+region_width] = recvbuf[r]

        plt.imshow(full_heatmap, cmap="hot")
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    main()
