from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        events = np.load("test_data/A62P20C3-2021_11_06_18_33_41.npy")
        N = events.shape

        chunks = np.array_split(events, size)
    else:
        chunks = None
        N = None

    N = comm.bcast(N, root=0)

    local_events = comm.scatter(chunks, root=0)

    print(f"Rank {rank}: received {len(local_events)} events")

    max_events = 5

    for i, event in enumerate(local_events[:max_events]):
        print(f"Rank {rank}: event[{i}] = {event}")

    comm.Barrier()
    
    if rank == 0:
        print("Finished, all is done.")


if __name__ == "__main__":
    main()
