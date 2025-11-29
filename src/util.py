import numpy as np
import numpy as np
import os

def get_dataset(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return np.load(path)

    elif ext == ".npz":
        data = np.load(path)

        required = ["x", "y", "t", "p"]
        for key in required:
            if key not in data:
                raise ValueError(f"NPZ file missing '{key}' array")

        x = data["x"]
        y = data["y"]
        t = data["t"]
        p = data["p"]

        events = np.column_stack((x, y, t, p))

        return events

    else:
        print("Wrong file format!")
        raise ValueError(f"Unsupported file extension for dataset: {ext}")
