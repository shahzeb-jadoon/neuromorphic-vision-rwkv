import h5py

file_path = "data/test_collection/episode_88f3d598.h5"
frames_to_check = [3650, 3735, 3739, 3748, 3752, 3763, 3776, 3783, 3790, 3798, 3830, 3831, 3850]

with h5py.File(file_path, "r") as f:
    for frame in frames_to_check:
        print(f"Frame {frame}:")
        # Find which dataset holds this data, could be 'metrics' or 'state' or 'actors'
        if "navigation" in f:
            for key in f["navigation"].keys():
                try:
                    val = f["navigation"][key][frame]
                    print(f"  {key}: {val}")
                except Exception:
                    pass
        if "state" in f:
            for key in f["state"].keys():
                try:
                    val = f["state"][key][frame]
                    print(f"  {key}: {val}")
                except Exception:
                    pass
        if "ego" in f:
            for key in f["ego"].keys():
                try:
                    val = f["ego"][key][frame]
                    print(f"  {key}: {val}")
                except Exception:
                    pass
