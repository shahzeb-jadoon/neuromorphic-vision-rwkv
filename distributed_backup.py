import os
import sys
import json
import hashlib
import subprocess
import time
import shutil

MANIFEST_FILE = "migration_state.json"
SOURCE_DIR = "/mnt/data/carla_data"


class UIManager:
    def __init__(self):
        self.lines = 8
        self.is_active = False
        self.mode = "backup"
        self.phase_names = [
            ("shash", "Source Hashing "),
            ("copy", "Compress & Copy"),
            ("thash", "Target Hashing "),
        ]

        # Drive info
        self.target_base = "Waiting..."
        self.free_space_gb = 0.0
        self.files_on_drive = 0
        self.est_files_on_drive = 1

        # Global info
        self.current_file_idx = 0
        self.total_files = 0
        self.current_filename = ""
        self.file_size_gb = 0.0

        # Averages
        self.avg_times = {"shash": 0.0, "copy": 0.0, "thash": 0.0}
        self.total_times = {"shash": 0.0, "copy": 0.0, "thash": 0.0}
        self.total_bytes = {"shash": 0, "copy": 0, "thash": 0}
        self.counts = {"shash": 0, "copy": 0, "thash": 0}

        # Live phase data (0.0 to 1.0), and speeds
        self.phases = {
            "shash": {"prog": 0.0, "speed": 0.0},
            "copy": {"prog": 0.0, "speed": 0.0},
            "thash": {"prog": 0.0, "speed": 0.0},
        }

        self.phase_start_time = 0
        self.phase_bytes_done = 0

    def set_mode(self, mode):
        self.mode = mode
        if mode == "restore":
            self.phase_names = [
                ("shash", "Check Existing "),
                ("copy", "Decompress Data"),
                ("thash", "Verify Target  "),
            ]
        elif mode == "fast_restore":
            self.phase_names = [("shash", "Check Existing "), ("copy", "Decompress Data")]
        else:
            self.phase_names = [
                ("shash", "Source Hashing "),
                ("copy", "Compress & Copy"),
                ("thash", "Target Hashing "),
            ]

    def start_dashboard(self):
        print("\n" * self.lines)
        self.is_active = True
        self.draw()

    def stop_dashboard(self):
        if self.is_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.is_active = False

    def _format_bar(self, percent):
        length = 18
        filled = int(length * percent)
        bar = "=" * filled + ">" + " " * (length - filled - 1)
        if filled == length:
            bar = "=" * length
        if percent <= 0:
            bar = " " * length
        return f"[{bar}] {percent*100:5.1f}%"

    def _format_time(self, secs):
        if secs <= 0:
            return "N/A"
        if secs < 60:
            return f"{int(secs)}s"
        return f"{int(secs//60)}m {int(secs%60)}s"

    def draw(self):
        if not self.is_active:
            return
        sys.stdout.write(f"\x1b[{self.lines}A")

        if self.mode == "backup":
            total_avg = self.avg_times["shash"] + self.avg_times["copy"] + self.avg_times["thash"]
            rem_files = max(0, self.est_files_on_drive - self.files_on_drive)
            est_time_left = rem_files * total_avg if total_avg > 0 else 0

            drive_pct = (
                self.files_on_drive / self.est_files_on_drive if self.est_files_on_drive > 0 else 0
            )
            drive_pct = min(1.0, drive_pct)

            sys.stdout.write(
                f"\x1b[2K[Drive: {self.target_base}] Free Space: {self.free_space_gb:.2f} GB\n"
            )
            sys.stdout.write(
                f"\x1b[2KDrive Capacity  : {self._format_bar(drive_pct)} "
                f"({self.files_on_drive} / ~{self.est_files_on_drive} Files) "
                f"| Est. Time Left: {self._format_time(est_time_left)}\n"
            )
        else:
            total_avg = sum(self.avg_times[p] for p in [p[0] for p in self.phase_names])
            rem_files = max(0, self.total_files - self.current_file_idx + 1)
            est_time_left = rem_files * total_avg if total_avg > 0 else 0

            mode_label = "Fast Restore" if self.mode == "fast_restore" else "Restore"
            sys.stdout.write(
                f"\x1b[2K[Target: {self.target_base}] Mode: {mode_label} "
                f"| Est. Time Left: {self._format_time(est_time_left)}\n"
            )
            sys.stdout.write("\x1b[2K\n")

        sys.stdout.write(
            f"\x1b[2KGlobal Progress : File {self.current_file_idx} / {self.total_files}\n"
        )
        sys.stdout.write("\x1b[2K\n")
        sys.stdout.write(
            f"\x1b[2K-- Current File: {self.current_filename} ({self.file_size_gb:.2f} GB) --\n"
        )

        for p, name in self.phase_names:
            d = self.phases[p]
            spd = f"{d['speed']:.1f} MB/s" if d["prog"] > 0 and d["prog"] < 1 else "---"
            avg_time = self._format_time(self.avg_times[p])

            avg_spd = "--- MB/s"
            if self.total_times[p] > 0:
                avg_spd_mbps = (self.total_bytes[p] / self.total_times[p]) / (1024**2)
                avg_spd = f"{avg_spd_mbps:.1f} MB/s"

            sys.stdout.write(
                f"\x1b[2K{name} : {self._format_bar(d['prog'])} "
                f"(Speed: {spd:>10} | Avg: {avg_time:>4} @ {avg_spd:>10})\n"
            )

        empty_lines = 3 - len(self.phase_names)
        for _ in range(empty_lines):
            sys.stdout.write("\x1b[2K\n")

        sys.stdout.flush()

    def set_drive_info(self, path, free_bytes, avg_file_size):
        self.target_base = path
        self.free_space_gb = free_bytes / (1024**3)
        req_per_file = avg_file_size * 1.05 if avg_file_size > 0 else (1024**3)
        self.est_files_on_drive = max(1, int(free_bytes / req_per_file))
        self.files_on_drive = 0
        if os.path.exists(path):
            self.files_on_drive = len([f for f in os.listdir(path) if f.endswith(".zst")])
        self.draw()

    def set_file_info(self, idx, total, name, size_gb):
        self.current_file_idx = idx
        self.total_files = total
        self.current_filename = name
        self.file_size_gb = size_gb
        for p in self.phases:
            self.phases[p] = {"prog": 0.0, "speed": 0.0}
        self.draw()

    def start_phase(self, phase):
        self.phase_start_time = time.time()
        self.phase_bytes_done = 0
        self.phases[phase]["prog"] = 0.0
        self.draw()

    def end_phase(self, phase):
        duration = time.time() - self.phase_start_time

        self.counts[phase] += 1
        self.total_times[phase] += duration
        self.total_bytes[phase] += self.phase_bytes_done
        self.avg_times[phase] = self.total_times[phase] / self.counts[phase]

        self.phases[phase]["prog"] = 1.0
        self.phases[phase]["speed"] = 0.0
        self.draw()

    def update_phase(self, phase, bytes_done, total_bytes):
        self.phase_bytes_done = bytes_done
        elapsed = time.time() - self.phase_start_time
        if elapsed > 0.5:
            speed = (bytes_done / (1024**2)) / elapsed
            self.phases[phase]["speed"] = speed
        if total_bytes > 0:
            self.phases[phase]["prog"] = min(1.0, bytes_done / total_bytes)
        self.draw()


ui = UIManager()


def get_hash(filepath, phase_name, chunk_size=8192 * 1024):
    ui.start_phase(phase_name)
    h = hashlib.sha256()
    total_size = os.path.getsize(filepath)
    processed = 0
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            processed += len(chunk)
            if total_size > 0:
                ui.update_phase(phase_name, processed, total_size)
    ui.end_phase(phase_name)
    return h.hexdigest()


def get_free_space(path):
    try:
        stat = os.statvfs(path)
        return stat.f_frsize * stat.f_bavail
    except Exception:
        return 0


def list_drives():
    print("\n--- Currently Mounted Drives ---")
    subprocess.run(["df", "-h", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs"], text=True)
    print("--------------------------------\n")


def auto_mount_helper():
    print("\n[Helper] Windows drives usually auto-mount to /mnt/c, /mnt/d, etc.")
    ans = (
        input("Do you need to manually mount a Windows drive that isn't showing up? (y/N): ")
        .strip()
        .lower()
    )
    if ans == "y":
        letter = input("Enter the Windows drive letter (e.g., E): ").strip().upper()
        if letter:
            target = f"/mnt/{letter.lower()}"
            print(f"Attempting to mount {letter}: to {target} ...")
            try:
                subprocess.run(["sudo", "mkdir", "-p", target], check=True)
                subprocess.run(["sudo", "mount", "-t", "drvfs", f"{letter}:", target], check=True)
                print(f"[+] Successfully mounted {letter}: to {target}")
            except Exception as e:
                print(f"[-] Failed to mount: {e}")
            time.sleep(1)


def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {"global_backup_counter": 1, "files": {}}


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)


def select_target_drive(required_space, filename, manifest, avg_file_size):
    ui.stop_dashboard()
    while True:
        list_drives()
        auto_mount_helper()
        print(f"\n[!] Next file: {filename}")
        print(f"[!] Required space: ~{required_space / (1024**3):.2f} GB")

        target_base = input(
            "\nEnter the base mount path (e.g., /mnt/e), 'rescan' to refresh, or 'exit': "
        ).strip()

        if target_base.lower() == "exit":
            print("Exiting safely. Your progress is saved.")
            sys.exit(0)
        elif target_base.lower() == "rescan" or not target_base:
            continue

        free_space = get_free_space(target_base)

        if free_space > required_space:
            existing = []
            if os.path.exists(target_base):
                existing = [
                    d
                    for d in os.listdir(target_base)
                    if d.startswith("carla_data_backup_")
                    and os.path.isdir(os.path.join(target_base, d))
                ]

            if existing:
                existing.sort(
                    key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
                )
                target_dir = os.path.join(target_base, existing[-1])
                print(f"\n[+] Resuming backup in existing folder: {target_dir}")
                subprocess.run(
                    ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", target_dir], check=True
                )
            else:
                session_id = manifest.get("global_backup_counter", 1)
                manifest["global_backup_counter"] = session_id + 1
                save_manifest(manifest)
                target_dir = os.path.join(target_base, f"carla_data_backup_{session_id}")
                print(f"\n[+] Creating new sequential backup folder: {target_dir}")
                subprocess.run(["sudo", "mkdir", "-p", target_dir], check=True)
                subprocess.run(
                    ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", target_dir], check=True
                )

            ui.set_drive_info(target_dir, free_space, avg_file_size)
            ui.start_dashboard()
            return target_dir, manifest
        else:
            print(
                f"\n[ERROR] Not enough space on {target_base}. "
                f"Has {free_space/(1024**3):.2f} GB, needs {required_space/(1024**3):.2f} GB."
            )
            input("Press Enter to try again...")


def backup_mode():
    if not shutil.which("zstd"):
        print("[ERROR] 'zstd' is not installed. Please run: sudo apt install zstd")
        sys.exit(1)

    print("\n=== STARTING DISTRIBUTED BACKUP ===")
    manifest = load_manifest()
    current_target_dir = None

    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory {SOURCE_DIR} not found.")
        sys.exit(1)

    files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    total_files = len(files)

    total_source_size = sum(os.path.getsize(os.path.join(SOURCE_DIR, f)) for f in files)
    avg_file_size = total_source_size / total_files if total_files > 0 else 0

    ui.start_dashboard()

    for idx, filename in enumerate(sorted(files), 1):
        try:
            filepath = os.path.join(SOURCE_DIR, filename)
            file_size = os.path.getsize(filepath)

            ui.set_file_info(idx, total_files, filename, file_size / (1024**3))

            if (
                filename in manifest["files"]
                and manifest["files"][filename].get("status") == "completed"
            ):
                continue

            orig_hash = get_hash(filepath, "shash")

            manifest["files"][filename] = {"original_hash": orig_hash, "status": "hashing"}
            save_manifest(manifest)

            required_space = int(file_size * 1.05)

            if current_target_dir is None or get_free_space(current_target_dir) < required_space:
                current_target_dir, manifest = select_target_drive(
                    required_space, filename, manifest, avg_file_size
                )

            compressed_path = os.path.join(current_target_dir, f"{filename}.zst")

            ui.start_phase("copy")
            chunk_size = 8192 * 1024
            processed = 0

            p = subprocess.Popen(
                ["zstd", "-T0", "-3", "-c"],
                stdin=subprocess.PIPE,
                stdout=open(compressed_path, "wb"),
            )
            with open(filepath, "rb") as f:
                while chunk := f.read(chunk_size):
                    p.stdin.write(chunk)
                    processed += len(chunk)
                    ui.update_phase("copy", processed, file_size)

            p.stdin.close()
            p.wait()
            ui.end_phase("copy")

            if p.returncode != 0:
                ui.stop_dashboard()
                print(f"\n[ERROR] Compression failed with code {p.returncode}")
                sys.exit(1)

            comp_hash = get_hash(compressed_path, "thash")

            manifest["files"][filename]["compressed_path"] = compressed_path
            manifest["files"][filename]["compressed_hash"] = comp_hash
            manifest["files"][filename]["status"] = "completed"
            save_manifest(manifest)

            ui.files_on_drive += 1
            ui.draw()

        except KeyboardInterrupt:
            ui.stop_dashboard()
            print("\n\n[!!!] Caught Ctrl+C (Safe Exit Requested) [!!!]")

            if (
                filename in manifest["files"]
                and manifest["files"][filename].get("status") != "completed"
            ):
                manifest["files"][filename]["status"] = "interrupted"
                save_manifest(manifest)

            if "compressed_path" in locals() and os.path.exists(compressed_path):
                try:
                    os.remove(compressed_path)
                    print(f"[*] Deleted incomplete backup chunk: {compressed_path}")
                except Exception:
                    pass

            print("[+] Cleanup complete. You can safely resume the script later!")
            sys.exit(0)

    ui.stop_dashboard()
    print("\n[SUCCESS] Backup Complete!")


def restore_mode(fast_mode=False):
    if not shutil.which("zstd"):
        print("[ERROR] 'zstd' is not installed. Please run: sudo apt install zstd")
        sys.exit(1)

    print("\n=== STARTING RESTORE ===")
    manifest = load_manifest()
    os.makedirs(SOURCE_DIR, exist_ok=True)

    # Pre-calculate files
    files_to_restore = [f for f, d in manifest["files"].items() if d.get("status") == "completed"]
    total_files = len(files_to_restore)

    avg_file_size = (
        sum(
            os.path.getsize(d["compressed_path"])
            for _, d in manifest["files"].items()
            if "compressed_path" in d and os.path.exists(d["compressed_path"])
        )
        / total_files
        if total_files > 0
        else 0
    )

    ui.set_mode("fast_restore" if fast_mode else "restore")
    ui.start_dashboard()
    ui.set_drive_info(SOURCE_DIR, get_free_space(SOURCE_DIR), avg_file_size)

    while True:
        all_completed = True
        processed_any_this_round = False

        for idx, filename in enumerate(sorted(files_to_restore), 1):
            data = manifest["files"][filename]

            if data.get("restored"):
                continue

            all_completed = False
            target_path = os.path.join(SOURCE_DIR, filename)
            comp_path = data["compressed_path"]

            ui.set_file_info(
                idx,
                total_files,
                filename,
                os.path.getsize(comp_path) / (1024**3) if os.path.exists(comp_path) else 0,
            )

            if os.path.exists(target_path):
                if get_hash(target_path, "shash") == data["original_hash"]:
                    manifest["files"][filename]["restored"] = True
                    save_manifest(manifest)
                    processed_any_this_round = True
                    continue

            if not os.path.exists(comp_path):
                continue

            processed_any_this_round = True

            try:
                # 1. Verify Compressed
                if not fast_mode:
                    if get_hash(comp_path, "shash") != data["compressed_hash"]:
                        ui.stop_dashboard()
                        print(f"\n[ERROR] Hash mismatch on {comp_path}! File is corrupted.")
                        sys.exit(1)

                # 2. Decompress & Copy
                ui.start_phase("copy")
                chunk_size = 8192 * 1024
                processed = 0
                comp_size = os.path.getsize(comp_path)

                p = subprocess.Popen(
                    ["zstd", "-d", "-T0", "-c"],
                    stdin=subprocess.PIPE,
                    stdout=open(target_path, "wb"),
                )
                with open(comp_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        p.stdin.write(chunk)
                        processed += len(chunk)
                        ui.update_phase("copy", processed, comp_size)

                p.stdin.close()
                p.wait()
                ui.end_phase("copy")

                if p.returncode != 0:
                    ui.stop_dashboard()
                    print(f"\n[ERROR] Decompression failed with code {p.returncode}")
                    sys.exit(1)

                # 3. Verify Target
                if not fast_mode:
                    if get_hash(target_path, "thash") == data["original_hash"]:
                        manifest["files"][filename]["restored"] = True
                        save_manifest(manifest)
                        ui.files_on_drive += 1
                        ui.draw()
                    else:
                        ui.stop_dashboard()
                        print(f"\n[ERROR] Final hash mismatch on {target_path}!")
                        sys.exit(1)
                else:
                    manifest["files"][filename]["restored"] = True
                    save_manifest(manifest)
                    ui.files_on_drive += 1
                    ui.draw()

            except KeyboardInterrupt:
                ui.stop_dashboard()
                print("\n\n[!!!] Caught Ctrl+C (Safe Exit Requested) [!!!]")
                if os.path.exists(target_path):
                    try:
                        os.remove(target_path)
                        print(f"[*] Deleted incomplete decompressed chunk: {target_path}")
                    except Exception:
                        pass
                print("[+] Cleanup complete. You can safely resume the script later!")
                sys.exit(0)

        if all_completed:
            ui.stop_dashboard()
            print("\n[SUCCESS] All files perfectly restored!")
            break

        if not processed_any_this_round:
            ui.stop_dashboard()
            print("\n[!] Please plug in the next backup drive (e.g., containing backup_2).")
            ans = input("Press Enter to rescan, or type 'exit' to quit: ").strip()
            if ans.lower() == "exit":
                sys.exit(0)
            auto_mount_helper()
            ui.start_dashboard()


def reset_mode():
    print("\n=== RESET BACKUP STATE ===")
    ans = input("Are you sure? This deletes 'migration_state.json'. (y/N): ").strip().lower()
    if ans == "y":
        if os.path.exists(MANIFEST_FILE):
            os.remove(MANIFEST_FILE)
            print("[+] Successfully deleted migration_state.json.")
        else:
            print("[!] No manifest file found.")
    else:
        print("Reset cancelled.")


if __name__ == "__main__":
    print("=========================================")
    print("  CARLA Distributed Migration Wizard")
    print("=========================================")
    print("1. Backup: Distribute G: to multiple drives")
    print("2. Restore: Reconstruct multiple drives to G: (Safe & Verified)")
    print("3. Fast Restore: Reconstruct without hashing (4x Faster)")
    print("4. Reset: Delete the save state to start over")

    choice = input("\nSelect mode (1, 2, 3, or 4): ").strip()
    if choice == "1":
        backup_mode()
    elif choice == "2":
        restore_mode(fast_mode=False)
    elif choice == "3":
        restore_mode(fast_mode=True)
    elif choice == "4":
        reset_mode()
    else:
        print("Invalid choice.")
