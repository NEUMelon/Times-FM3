import os
import sys
import posixpath
import paramiko
import time
from typing import Optional

REMOTE_HOST = os.environ.get("AUTODL_HOST", "connect.autodl.com")
REMOTE_PORT = int(os.environ.get("AUTODL_PORT", "22"))
REMOTE_USER = os.environ.get("AUTODL_USER", "root")
REMOTE_PASS = os.environ.get("AUTODL_PASS", "")
REMOTE_PROJECT_DIR = os.environ.get("AUTODL_PROJECT_DIR", "/root/autodl-tmp/DASFAA")

def get_ssh_client(retries: int = 3, delay: float = 2.0) -> paramiko.SSHClient:
    last_err = None
    for attempt in range(retries):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=REMOTE_HOST,
                port=REMOTE_PORT,
                username=REMOTE_USER,
                password=REMOTE_PASS,
                timeout=20,
                banner_timeout=35
            )
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(15)
            return client
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_err

def run_remote(cmd: str, env_turbo: bool = False, print_output: bool = True) -> int:
    """Run command remotely inside bash login shell."""
    client = get_ssh_client()
    try:
        turbo_prefix = "source /etc/network_turbo 2>/dev/null; " if env_turbo else ""
        export_pythonpath = f"export PYTHONPATH={REMOTE_PROJECT_DIR}:$PYTHONPATH; "
        full_cmd = f"bash -lc '{turbo_prefix}{export_pythonpath}cd {REMOTE_PROJECT_DIR} 2>/dev/null || cd /root; {cmd}'"
        if print_output:
            print(f">>> [REMOTE] {cmd}")
        
        stdin, stdout, stderr = client.exec_command(full_cmd, get_pty=False)
        stdin.close()
        
        for line in iter(stdout.readline, ""):
            if print_output:
                sys.stdout.write(line)
                sys.stdout.flush()
                
        err_output = stderr.read().decode('utf-8', errors='replace').strip()
        if err_output and print_output:
            lines = [l for l in err_output.split("\n") if "AutoDL" not in l and "turbo" not in l]
            if lines:
                print("[STDERR]:\n" + "\n".join(lines))
                
        exit_status = stdout.channel.recv_exit_status()
        return exit_status
    finally:
        client.close()

def launch_bg(cmd: str, log_name: str = "run.log") -> int:
    """Launch command in background using nohup and return immediately."""
    client = get_ssh_client()
    try:
        log_path = f"/root/autodl-tmp/results/{log_name}"
        export_pythonpath = f"export PYTHONPATH={REMOTE_PROJECT_DIR}:$PYTHONPATH; "
        bg_cmd = f"bash -lc 'mkdir -p /root/autodl-tmp/results; {export_pythonpath}cd {REMOTE_PROJECT_DIR}; nohup {cmd} > {log_path} 2>&1 & echo $!'"
        stdin, stdout, stderr = client.exec_command(bg_cmd, get_pty=False)
        stdin.close()
        pid = stdout.read().decode().strip()
        print(f">>> [BG LAUNCHED] PID: {pid} | Log: {log_path}")
        return 0
    finally:
        client.close()

def tail_log(log_name: str = "run.log", lines: int = 40):
    """Tail log from remote."""
    log_path = f"/root/autodl-tmp/results/{log_name}"
    run_remote(f"tail -n {lines} {log_path} 2>/dev/null || echo 'Log file not found yet.'")

def sync_file_to_remote(local_file: str, remote_file: str, retries: int = 3):
    """Upload a single file to remote with retry."""
    last_err = None
    for attempt in range(retries):
        client = None
        try:
            client = get_ssh_client()
            sftp = client.open_sftp()
            remote_dir = posixpath.dirname(remote_file)
            _sftp_mkdirs(sftp, remote_dir)
            sftp.put(local_file, remote_file)
            sftp.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(2.0)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
    raise last_err

def sync_dir_to_remote(local_dir: str, remote_dir: str, ignore_patterns=None):
    """Recursively upload a local directory to remote."""
    if ignore_patterns is None:
        ignore_patterns = [
            ".git", "__pycache__", ".pytest_cache", ".idea", ".vscode", "cache",
            "results", "dasfaa_claude_package", "LaTeX2e+Proceedings+Template+ZIP",
            "paper", "paper_figures", "analysis", "CLaude Review"
        ]
        
    client = get_ssh_client()
    try:
        sftp = client.open_sftp()
        for root, dirs, files in os.walk(local_dir):
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            rel_path = os.path.relpath(root, local_dir).replace("\\", "/")
            curr_remote_dir = posixpath.join(remote_dir, rel_path) if rel_path != "." else remote_dir
            _sftp_mkdirs(sftp, curr_remote_dir)
            
            for file in files:
                if any(p in file for p in ignore_patterns) or file.endswith((".zip", ".tar.gz", ".png", ".pdf")):
                    continue
                local_f = os.path.join(root, file)
                remote_f = posixpath.join(curr_remote_dir, file)
                sftp.put(local_f, remote_f)
                print(f"  Synced: {rel_path}/{file}")
        sftp.close()
    finally:
        client.close()

def download_file_from_remote(remote_file: str, local_file: str):
    """Download a single file from remote."""
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    client = get_ssh_client()
    try:
        sftp = client.open_sftp()
        sftp.get(remote_file, local_file)
        sftp.close()
    finally:
        client.close()

def _sftp_mkdirs(sftp, remote_directory):
    if remote_directory in ("", "/", "."):
        return
    try:
        sftp.stat(remote_directory)
    except IOError:
        parent = posixpath.dirname(remote_directory)
        _sftp_mkdirs(sftp, parent)
        try:
            sftp.mkdir(remote_directory)
        except IOError:
            pass

def fast_sync():
    """Package code folders into a temporary tar.gz, upload and extract on remote."""
    import tarfile
    import tempfile
    
    local_dir = os.path.dirname(os.path.abspath(__file__))
    tar_path = os.path.join(tempfile.gettempdir(), "dasfaa_sync.tar.gz")
    
    folders_to_sync = ["probe", "experiments", "tests", "scripts"]
    files_to_sync = ["remote_client.py"]
    
    print(f"Creating archive for folders: {folders_to_sync} ...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for folder in folders_to_sync:
            fpath = os.path.join(local_dir, folder)
            if os.path.exists(fpath):
                def filter_fn(tarinfo):
                    if any(ign in tarinfo.name for ign in ["__pycache__", ".pytest_cache", ".pyc"]):
                        return None
                    return tarinfo
                tar.add(fpath, arcname=folder, filter=filter_fn)
        for f in files_to_sync:
            fpath = os.path.join(local_dir, f)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=f)
                
    tar_size_kb = os.path.getsize(tar_path) / 1024
    print(f"Archive created ({tar_size_kb:.1f} KB). Uploading to remote...")
    remote_tar = "/tmp/dasfaa_sync.tar.gz"
    sync_file_to_remote(tar_path, remote_tar)
    try:
        os.remove(tar_path)
    except Exception:
        pass
    
    print("Extracting on remote...")
    cmd = f"mkdir -p {REMOTE_PROJECT_DIR} && tar -xzf {remote_tar} -C {REMOTE_PROJECT_DIR}/ && rm -f {remote_tar}"
    ret = run_remote(cmd, print_output=True)
    if ret == 0:
        print(">>> Fast sync SUCCESSFUL!")
    else:
        print(f">>> Fast sync failed with code {ret}")
    return ret

def main():
    if len(sys.argv) < 2:
        print("Usage: python remote_client.py [status | run <cmd> | bg <cmd> [log_name] | tail [log_name] | script <path> | sync]")
        sys.exit(1)
        
    action = sys.argv[1]
    
    if action == "status":
        run_remote("pwd; nvidia-smi; ps aux | grep python3 | grep -v grep | head -n 10")
    elif action == "run":
        cmd = " ".join(sys.argv[2:])
        ret = run_remote(cmd)
        sys.exit(ret)
    elif action == "bg":
        cmd = sys.argv[2]
        log = sys.argv[3] if len(sys.argv) > 3 else "run.log"
        launch_bg(cmd, log)
    elif action == "tail":
        log = sys.argv[2] if len(sys.argv) > 2 else "run.log"
        lines = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        tail_log(log, lines)
    elif action == "script":
        script_path = sys.argv[2]
        if not os.path.exists(script_path):
            print(f"Error: script not found: {script_path}")
            sys.exit(1)
        ext = os.path.splitext(script_path)[1]
        remote_tmp = f"/tmp/run_{os.getpid()}{ext}"
        sync_file_to_remote(script_path, remote_tmp)
        runner = "python3" if ext == ".py" else "bash"
        ret = run_remote(f"{runner} {remote_tmp}")
        run_remote(f"rm -f {remote_tmp}", print_output=False)
        sys.exit(ret)
    elif action == "sync":
        ret = fast_sync()
        sys.exit(ret)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

if __name__ == "__main__":
    main()
