import numpy as np

# a simple function that creates a heatmap of events
def process_heatmap(local_data, heatmap, x_start, y_start, width, height):
    for (x, y, t, p) in local_data:
        lx = int(x - x_start)
        ly = int(y - y_start)
        if 0 <= lx < width and 0 <= ly < height:
            heatmap[ly, lx] += 1

# background filtering algorithm based on the filter described in Delbruck's Frame Free Dynamic Digital Vision
# For each event, we update its 8 surrounding pixels with the last t while testing whether it meets a threshold
def process_background_filter(local_data, local_timestamp, x_start, y_start, width, height, T_thresh):

    output_events = []

    for x, y, t, p in local_data:
        lx = int(x - x_start)
        ly = int(y - y_start)

        if not (0 <= lx < width and 0 <= ly < height):
            continue

        # save last timestamp at pixel
        prev_t = local_timestamp[ly, lx]

        # threshold t value
        if t - prev_t <= T_thresh:
            output_events.append((x, y, t, p))

        # update 8 surrounding neighbors with t-val
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = lx + dx
                ny = ly + dy
                if 0 <= nx < width and 0 <= ny < height:
                    local_timestamp[ny, nx] = t

    if len(output_events) == 0:
        return np.empty((0, 4), dtype=np.float64)

    return np.array(output_events, dtype=np.float64)

# based on Sironi's HATS (Histograms of Averaged Time Surfaces) method
# creates a time surface and then calculates a local average
def process_hats_descriptor(local_data, time_surface, hats, width, height, tau=0.02, local_timestamp=0.0, block_w=32, block_h=32):
    for x, y, t, p in local_data:
        ix = int(x)
        iy = int(y)
        if 0 <= x < width and 0 <= y < height:
            time_surface[iy, ix] = np.exp(-(local_timestamp - t)/tau)

    # compute block averages
    ny, nx = hats.shape
    for by in range(ny):
        for bx in range(nx):
            patch = time_surface[by*block_h:(by+1)*block_h, bx*block_w:(bx+1)*block_w]
            hats[by, bx] = patch.mean()