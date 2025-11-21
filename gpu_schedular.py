import subprocess
import time
import os
from datetime import datetime

try:
    from colorama import init, Fore, Style

    init(autoreset=True)
except ImportError:
    # fallback if colorama not installed
    class Dummy:
        def __getattr__(self, item):
            return ""

    Fore = Style = Dummy()

COMMAND_FILE = "exp_5_scenes.sh"
POLL_INTERVAL = 30  # seconds between checks
BATCH_MARKER = "###"
LOG_FILE = "launcher_log.txt"
VISIBLE_GPU_IDS = [1, 2, 3, 4, 5]  # specify which GPUs to consider
MIN_MEMORY_MB_PER_GPU = 23000  # minimum free memory (in MB) required to use a GPU


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_command_batches(filepath):
    """Parses a single file into multiple command batches."""
    batches = []
    current_batch = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or (line.startswith("#") and not line.startswith(BATCH_MARKER)):
                continue  # skip empty or regular comment lines
            if line.startswith(BATCH_MARKER):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                if line.strip() != BATCH_MARKER:
                    print(
                        f"{Fore.YELLOW}⚠️ Warning: Unexpected batch marker format: {line}"
                    )
            else:
                current_batch.append(line)
        if current_batch:
            batches.append(current_batch)

    return batches


def get_free_gpus(min_memory_mb=MIN_MEMORY_MB_PER_GPU):
    """Returns a list of GPU IDs that have at least min_memory_mb available."""
    try:
        output = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
                shell=True,
            )
            .decode("utf-8")
            .strip()
        )

        gpu_stats = []
        for line in output.split("\n"):
            idx, mem_free = line.split(",")
            gpu_stats.append((int(idx.strip()), int(mem_free.strip())))

        # Filter GPUs that have enough free memory
        eligible_gpus = [
            idx for idx, mem_free in gpu_stats if mem_free >= min_memory_mb
        ]

        # Sort by most free memory first
        eligible_gpus = sorted(
            eligible_gpus,
            key=lambda idx: next(mem for i, mem in gpu_stats if i == idx),
            reverse=True,
        )
        eligible_gpus = [idx for idx in eligible_gpus if idx in VISIBLE_GPU_IDS]
        print(f"{Fore.GREEN}🖥️  Free GPUs: {eligible_gpus}")

        return eligible_gpus

    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Error querying GPU memory: {e}")
        return []


def run_on_gpu(command, gpu_id):
    """Launch a process with a specified GPU using CUDA_VISIBLE_DEVICES."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    print(f"{Fore.CYAN}[Launching] GPU {gpu_id}: {command}")

    # Log to file
    with open(LOG_FILE, "a") as log_f:
        log_f.write(f"[{timestamp()}] [LAUNCH] GPU {gpu_id}: {command}\n")

    return subprocess.Popen(command, shell=True, env=env)


def run_batch(commands):
    running_procs = []
    success_count = 0
    failure_count = 0

    while commands or running_procs:
        prev_running = len(running_procs)

        finished_procs = [p for p in running_procs if p.poll() is not None]
        running_procs = [p for p in running_procs if p.poll() is None]

        for p in finished_procs:
            exit_code = p.returncode
            if exit_code == 0:
                success_count += 1
            else:
                failure_count += 1
                with open(LOG_FILE, "a") as log_f:
                    log_f.write(
                        f"[{timestamp()}] ❌ Command failed (exit {exit_code}): {p.args}\n"
                    )
                if exit_code == -9:
                    print(f"{Fore.RED}💥 Process killed (probably OOM or manual kill)!")
                else:
                    print(f"{Fore.RED}❌ Command failed (exit {exit_code})")

        if len(running_procs) < prev_running or not running_procs:
            free_gpus = get_free_gpus()

            while free_gpus and commands:
                gpu_id = free_gpus.pop(0)
                cmd = commands.pop(0)
                try:
                    proc = run_on_gpu(cmd, gpu_id)
                    running_procs.append(proc)
                    time.sleep(40)  # Small gap between launches
                except Exception as e:
                    print(f"{Fore.RED}❌ Failed to launch command: {cmd}\nError: {e}")
                    with open(LOG_FILE, "a") as log_f:
                        log_f.write(
                            f"[{timestamp()}] ❌ Failed to launch: {cmd}\nError: {e}\n"
                        )
                    failure_count += 1

        time.sleep(POLL_INTERVAL)

    return success_count, failure_count


def main():
    batches = parse_command_batches(COMMAND_FILE)
    print(f"{Fore.MAGENTA}📦 Found {len(batches)} batches in {COMMAND_FILE}")

    total_success = 0
    total_failure = 0

    for idx, batch in enumerate(batches, 1):
        print(f"\n{Fore.GREEN}🚀 Starting Batch {idx} ({len(batch)} commands)")
        success, failure = run_batch(batch)
        total_success += success
        total_failure += failure
        print(
            f"{Fore.GREEN}✅ Finished Batch {idx}: {success} succeeded, {failure} failed"
        )

    print(f"\n{Fore.BLUE}🎉 All batches completed.")
    print(
        f"{Fore.GREEN}🏁 Final Summary: {total_success} succeeded, {Fore.RED}{total_failure} failed"
    )

    with open(LOG_FILE, "a") as log_f:
        log_f.write(
            f"\n🏁 Final Summary: {total_success} succeeded, {total_failure} failed\n"
        )


if __name__ == "__main__":
    main()
# This script is designed to be run in a bash environment with access to nvidia-smi.
