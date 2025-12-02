import re
import numpy as np

RUN_HEADER_RE = re.compile(
    r"Running with (\d+)(?:\s+\((\d+)x(\d+)\))? processes"
)
ITERATION_RE = re.compile(r"Iteration\s+(\d+)")
TIME_RE = re.compile(
    r"\[TIME\]\s+(MPI Heatmap|MPI Background Filtering|MPI HATS|"
    r"Serial Heatmap|Serial Background Filtering|Serial HATS descriptor)"
    r"\s+time:\s+([0-9]*\.[0-9]+)"
)

def make_run_label(proc, pw, ph):
    if proc == "serial":
        return "Serial"
    if pw and ph:
        return f"{proc}({pw}x{ph})"
    return f"{proc}"

def parse_benchmark_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    data = {}  # data[(category, run_label)][test] = list
    current_iteration = None
    current_run = None  # like "64", "32(8x4)", "serial"
    temp_counts = {}     # count per (category,run_label) per iteration

    for line in lines:

        # detect iteration
        it = ITERATION_RE.search(line)
        if it:
            current_iteration = int(it.group(1))
            temp_counts = {}
            current_run = None
            continue

        # detect "Running with ..." header
        rh = RUN_HEADER_RE.search(line)
        if rh:
            proc = rh.group(1)
            pw = rh.group(2)
            ph = rh.group(3)
            current_run = make_run_label(proc, pw, ph)
            continue

        # detect serial block
        if "Running serial" in line:
            current_run = "serial"
            continue

        # detect time entry
        tm = TIME_RE.search(line)
        if tm:
            category = tm.group(1)
            value = float(tm.group(2))

            # full key includes runtime configuration
            key = (category, current_run)

            if key not in data:
                data[key] = {1: [], 2: []}

            if key not in temp_counts:
                temp_counts[key] = 0

            temp_counts[key] += 1
            c = temp_counts[key]

            if c not in (1, 2):
                raise RuntimeError(
                    f"More than 2 tests found for {key} in iteration {current_iteration}"
                )

            data[key][c].append(value)

    for key in data:
        for t in (1, 2):
            data[key][t] = np.array(data[key][t], dtype=np.float64)

    return data


if __name__ == "__main__":
    log_path = "DV_MPI_TEST_149984.out"
    results = parse_benchmark_file(log_path)

    for key, tests in results.items():
        print(f"{key} Small:", tests[1])
        print(f"{key} Large:", tests[2])
        print()
