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

def process_hats_descriptor(events, width, height, block_size=16, rho=2, tau=0.03):
    n_cells_y = (height + block_size - 1) // block_size
    n_cells_x = (width + block_size - 1) // block_size

    # init grid to store the last time stamp per pixel
    last_timestamp = np.ones((height, width), dtype=np.float64)

    # init local hats
    hats = np.zeros((n_cells_y, n_cells_x, 2), dtype=np.float64)

    # init local count per pixel, used for normalization
    counts = np.zeros((n_cells_y, n_cells_x, 2), dtype=np.int64)

    for x, y, t, p in events:
        ix = int(x)
        iy = int(y)
        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            continue

        # update memory for this pixel
        last_timestamp[iy, ix] = t

        # init local time surface
        local_time_surface = np.zeros((2 * rho + 1, 2 * rho + 1), dtype=np.float32)

        for dy in range(-rho, rho + 1):
            for dx in range(-rho, rho + 1):
                yy = iy + dy
                xx = ix + dx
                if 0 <= xx < width and 0 <= yy < height:
                    dt = t - last_timestamp[yy, xx]
                    if dt >= 0:
                        local_time_surface[dy + rho, dx + rho] = np.exp(-dt / tau)
                    else:
                        local_time_surface[dy + rho, dx + rho] = 0.0

        # map pixel to cell
        cell_x = int(ix // block_size)
        cell_y = int(iy // block_size)
        ip = int(p)

        hats[cell_y, cell_x, ip] += local_time_surface.sum()
        counts[cell_y, cell_x, ip] += 1

    # normalize
    nonzero = counts > 0
    hats[nonzero] /= counts[nonzero]

    return hats