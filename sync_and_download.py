import os
import stat
import sys
from pathlib import Path
import paramiko

REMOTE_HOST = "connect.bjb1.seetacloud.com"
REMOTE_PORT = 11667
REMOTE_USER = "root"
REMOTE_PASS = "WLRJRlKe27Bk"

LOCAL_BASE = Path(r"E:\Desktop\DASFAA")
LOCAL_RESULTS = LOCAL_BASE / "results"
LOCAL_FIGURES = LOCAL_BASE / "paper_figures"

LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
LOCAL_FIGURES.mkdir(parents=True, exist_ok=True)

def download_folder(sftp, remote_dir, local_dir):
    print(f"\nScanning remote directory: {remote_dir} ...")
    try:
        items = sftp.listdir_attr(remote_dir)
    except Exception as e:
        print(f"Error accessing {remote_dir}: {e}")
        return

    for item in items:
        # Skip subdirectories like .ipynb_checkpoints
        if stat.S_ISDIR(item.st_mode):
            print(f"  [SKIP DIR] {item.filename}")
            continue

        remote_file = f"{remote_dir}/{item.filename}"
        local_file = local_dir / item.filename
        try:
            print(f"  Downloading {item.filename} ({item.st_size} bytes) ...")
            sftp.get(remote_file, str(local_file))
            print(f"    [DONE] -> {local_file}")
        except Exception as e:
            print(f"    [FAIL] {item.filename}: {e}")

def main():
    print(f"Connecting to {REMOTE_HOST}:{REMOTE_PORT} via SFTP...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASS)
    sftp = ssh.open_sftp()

    # 1. Download results
    download_folder(sftp, "/root/autodl-tmp/results", LOCAL_RESULTS)

    # 2. Download paper figures and tables
    download_folder(sftp, "/root/autodl-tmp/DASFAA/paper_figures", LOCAL_FIGURES)

    sftp.close()
    ssh.close()
    print("\nAll downloads successfully completed!")

if __name__ == "__main__":
    main()
