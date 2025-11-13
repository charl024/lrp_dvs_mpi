import numpy as np

events = np.load("./THU-EACT-50-CHL/A62P20C3-2021_11_06_18_33_41.npy")
batch_size = 10000
for i in range(0, len(events), batch_size):
    batch = events[i:i+batch_size]
    for x, y, t, p in batch:
        print(t)
