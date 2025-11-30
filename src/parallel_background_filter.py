from mpi4py import MPI
import numpy as np
import processing as prc
import matplotlib.pyplot as plt
import time

def transform_to_heatmap(filtered_data, width, height):
    filtered_heatmap = np.zeros((height, width), dtype=np.int32)

    xs = filtered_data[:, 0].astype(int)
    ys = filtered_data[:, 1].astype(int)

    mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[mask]
    ys = ys[mask]

    for x, y in zip(xs, ys):
        filtered_heatmap[y, x] += 1

    return filtered_heatmap

def parallel_background_filter(events, width, height, process_width, process_height, packet_size=1000, T_thresh=2000.0, display=False):    
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

    # print(f"Rank {rank} owns region x[{x_start}:{x_end}], y[{y_start}:{y_end}]")

    # grid of local timestamps across all ranks
    local_timestamp = np.full((region_height, region_width), -1.0, dtype=np.float64)
    # store locally filtered events across all ranks
    local_filtered = []

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
            out_r0 = prc.process_background_filter(
                local_data=local_data_r0,
                local_timestamp=local_timestamp,
                x_start=x_start,
                y_start=y_start,
                width=region_width,
                height=region_height,
                T_thresh=T_thresh
            )

            local_filtered.append(out_r0)

            

        for r in range(1, size):
            comm.send(None, dest=r, tag=1)

        # print(f"Rank 0: processed {local_count} events locally")

    else:

        while True:
            status = MPI.Status()
            local_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            current_tag = status.Get_tag()

            if current_tag == 1:
                # print(f"Rank {rank}: received end of data stream signal")
                break

            # process one packet
            out = prc.process_background_filter(
                local_data=local_data,
                local_timestamp=local_timestamp,
                x_start=x_start,
                y_start=y_start,
                width=region_width,
                height=region_height,
                T_thresh=T_thresh
            )

            local_filtered.append(out)

    # gather local filtered events
    if local_filtered:
        sendbuf = np.concatenate(local_filtered, axis=0)
    else:
        sendbuf = np.zeros((0,4), dtype=np.float64)

    all_filtered = comm.gather(sendbuf, root=0)

    comm.Barrier()
    end_time = MPI.Wtime()

    # calculate total time taken to process, create full_filtered_heatmap and then display
    if rank == 0:
        filtered = np.concatenate(all_filtered, axis=0)

        total_time = end_time - start_time
        print(f"[TIME] MPI Background Filtering time: {total_time:.6f} seconds")

        if not display:
            return

        full_filtered_heatmap = transform_to_heatmap(filtered, width, height)
        full_filtered_heatmap_log = np.log1p(full_filtered_heatmap)
        # visualization
        plt.imshow(
            filtered_heatmap_log,
            cmap="hot",
            extent=[0, width, height, 0]
        )

        plt.colorbar(label="Event Density")
        plt.title("Filtered Event Heatmap")
        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.tight_layout()
        plt.savefig("plots/parallel/mpi_background_filter_out.svg")