from mpi4py import MPI
import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

def parallel_hats(events, width, height, process_width, process_height, block_size=32, packet_size=1000, tau=0.02, rho=2, display=False):
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

    # init local hats 
    n_cells_y = (height + block_size - 1) // block_size
    n_cells_x = (width  + block_size - 1) // block_size
    local_hats = np.zeros((n_cells_y, n_cells_x, 2), dtype=np.float64)

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

            # processing on rank 0
            local_data_r0 = np.array(buckets[0], dtype=packet.dtype)

            hats_patch = prc.process_hats_descriptor(
                events=local_data_r0,
                width=width,
                height=height,
                block_size=block_size,
                rho=rho,
                tau=tau
            )

            local_hats += hats_patch


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

            # process one packet using hats
            hats_patch = prc.process_hats_descriptor(
                events=local_data,
                width=width,
                height=height,
                block_size=block_size,
                rho=rho,
                tau=tau
            )

            local_hats += hats_patch

    # reduce all local hats in rank 0

    if rank == 0:
        full_hats = np.zeros_like(local_hats, dtype=np.float64)
    else:
        full_hats = None

    comm.Reduce(local_hats, full_hats, op=MPI.SUM, root=0)

    comm.Barrier()
    end_time = MPI.Wtime()


    # calculate total time taken to process, create full_hats and then display
    if rank == 0:
        total_time = end_time - start_time
        print(f"[TIME] MPI HATS time: {total_time:.6f} seconds")

        if not display:
            return

        # form full hats map here

        

        # visualization
        hats_pos = full_hats[:, :, 1]
        hats_neg = full_hats[:, :, 0]
        hats_sum = hats_pos + hats_neg

        plt.figure(figsize=(8, 6))
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
        plt.savefig("plots/parallel/mpi_hats_out.svg")