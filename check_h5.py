import h5py


def check_h5(filepath):
    try:
        with h5py.File(filepath, "r") as f:
            print(f"File: {filepath}")

            def print_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(
                        f"Dataset: {name}, Shape: {obj.shape}, "
                        f"Type: {obj.dtype}, Compression: {obj.compression}"
                    )

            f.visititems(print_info)
    except Exception as e:
        print(f"Error: {e}")


check_h5("/mnt/data/carla_data/episode_038d6375.h5")
