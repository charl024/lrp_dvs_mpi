import numpy as np

# a simple function that creates a heatmap of events
def process_heatmap(local_data, heatmap, x_start, y_start, width, height):
    for (x, y, t, p) in local_data:
        lx = int(x - x_start)
        ly = int(y - y_start)
        if 0 <= lx < width and 0 <= ly < height:
            heatmap[ly, lx] += 1

# background filtering algorithm based on the filter described in Frame Free Dynamic Digital Vision
def process_background_filter(local_data, local_timestamp, x_start, y_start, width, height, T_thresh):
    output_events = []
    for x, y, t, p in local_data:
        lx = int(x - x_start)
        ly = int(y - y_start)

        if not (0 <= lx < width and 0 <= ly < height):
            continue

        # neighbor update
        for dx in (-1, 0, +1):
            for dy in (-1, 0, +1):
                nx = int(lx + dx)
                ny = int(ly + dy)
                if 0 <= nx < width and 0 <= ny < height:
                    local_timestamp[ny, nx] = t

        # check survival
        if t - local_timestamp[ly, lx] <= T_thresh:
            output_events.append((x, y, t, p))

    return output_events

# based on HATS
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