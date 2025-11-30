from mpi4py import MPI
import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

def parallel_heatmap(events, width, height, process_width, process_height, packet_size=1000, display=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    region_width  = width // process_width
    region_height = height // process_height

    if process_width * process_height != size:
        print("Error: Num Processes must equal ", process_width * process_height)
        return

    rx = rank % process_width
    ry = rank // process_width

    x_start = rx * region_width
    x_end = (rx + 1) * region_width if rx != process_width - 1 else width

    y_start = ry * region_height
    y_end = (ry + 1) * region_height if ry != process_height - 1 else height

    # init local heatmap
    local_heatmap = np.zeros((region_height, region_width), dtype=np.int32)

    comm.Barrier()
    start_time = MPI.Wtime()

    if rank == 0:
        # simulate packet stream
        for i in range(0, len(events), packet_size):
            packet = events[i:i+packet_size]

            buckets = {r: [] for r in range(size)}

            for x, y, t, p in packet:
                tx = min(x // region_width,  process_width - 1)
                ty = min(y // region_height, process_height - 1)
                r  = ty * process_width + tx
                buckets[r].append((x, y, t, p))

            # send buckets to other ranks
            for r in range(1, size):
                bucket_arr = np.array(buckets[r], dtype=packet.dtype)
                comm.send(bucket_arr, dest=r, tag=0)
                # print(f"Rank 0: sent {len(buckets[r])} events to rank {r}")

            # processing on rank 0
            local_data_r0 = np.array(buckets[0], dtype=packet.dtype)

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

    else:

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
        print(f"[TIME] MPI Heatmap time: {total_time:.6f} seconds")

        full_heatmap = np.zeros((height, width), dtype=np.int32)

        if not display:
            return

        for r in range(size):
            rx = r % process_width
            ry = r // process_width

            x0 = rx * region_width
            y0 = ry * region_height

            full_heatmap[y0:y0+region_height, x0:x0+region_width] = recvbuf[r]

        # visualization
        plt.figure(figsize=(8, 6))
        heatmap_log = np.log1p(full_heatmap)

        plt.imshow(
            heatmap_log,
            cmap="hot",
            extent=[0, width, 0, height]
        )

        plt.colorbar(label="Event Density")
        plt.title("Event Heatmap")
        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.tight_layout()
        plt.savefig("plots/parallel/mpi_heatmap_out.svg")