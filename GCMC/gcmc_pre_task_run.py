'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 15:15:58
'''
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from GCMC.utils import run_simulation
from concurrent.futures import ProcessPoolExecutor
import argparse

def run_subprocess(task_dir):
        process = run_simulation(task_dir, executor="bash", script_name="run")
        stdout, stderr = process.communicate() 
        return stdout, stderr

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GCMC pre-task run')
    parser.add_argument('--workdir', type=str, help='Working directory', required=True)
    parser.add_argument('--n_cpus', type=int, default=300, help='Number of CPUs to use', required=False)
    args = parser.parse_args()

    ## run pre-tasks
    workdir = Path(args.workdir)
    task_dirs = []
    for sub_dir in workdir.glob("*"):
        if not sub_dir.is_dir():
            continue
        charge_dir = sub_dir/"_charge_eq"
        if (not charge_dir.exists()) or (charge_dir/"Movies/System_0/").exists():
            continue
        task_dirs.append( charge_dir)
        vf_dir = sub_dir/"_helium_void_fraction"
        if (not vf_dir.exists()) or (vf_dir/"Output/System_0/").exists():
            continue
        task_dirs.append(vf_dir)
        
    num_parallel_processes = min(args.n_cpus, len(task_dirs))
    print(f"Running {len(task_dirs)} tasks with {num_parallel_processes} processes")
    with ProcessPoolExecutor(max_workers=num_parallel_processes) as executor:
        results = list(executor.map(run_subprocess, task_dirs))
    for (stdout, stderr), task_dir in zip(results, task_dirs):
        with open(task_dir/"gcmc.out", "w") as f:
            f.write(stdout.decode("utf-8"))
        with open(task_dir/"gcmc.err", "w") as f:
            f.write(stderr.decode("utf-8"))

    