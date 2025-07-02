'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 16:31:28
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import shutil
from GCMC.gcmc_process import create_tasks, run_and_check_pre_tasks
from GCMC.utils import run_simulation
import time
import random
from decimal import Decimal
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse
import json

def run_subprocess(task_dir):
        process = run_simulation(task_dir, executor="bash", script_name="run")
        stdout, stderr = process.communicate() 
        return stdout, stderr

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GCMC pre-task run')
    parser.add_argument('--workdir', type=str, default=None, help='Working directory', required=True)
    parser.add_argument('--n_cpus', type=int, default=300, help='Number of CPUs to use', required=False)
    parser.add_argument('--max_press', type=int, default=10000000, help='Maximum pressure to run', required=False)
    parser.add_argument('--min_press', type=int, default=0, help='Minimum pressure to run', required=False)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    print("workdir: {}".format(workdir))

    ## run pre-tasks
    task_dirs = []
    # select_name_list = [n.name for n in workdir.glob("*") if n.is_dir()]
    with open(workdir/"00_gcmc_params.json", "r") as f:
        user_defined_params = json.load(f)

    task_tuples = user_defined_params["TaskTuples"]
    unfinished_dirs_charge_eq, unfinished_dirs_void_fraction = run_and_check_pre_tasks(workdir, task_tuples, interval=60,
                            num_interval = 1,
                            executor = "bash",
                            check_only = True)
    exclude_dirs = set([t[0].name for t in unfinished_dirs_void_fraction + unfinished_dirs_charge_eq])
    for sub_dir in workdir.glob("*"):
        if not sub_dir.is_dir() or (sub_dir in exclude_dirs) or (sub_dir.name in exclude_dirs):
            continue
        for task_name, press in task_tuples:
            if press > args.max_press or press <= args.min_press:
                continue
            if len(list((sub_dir/task_name/"Movies/System_0").glob("Framework_0_final*"))) > 0:
                continue
            task_dirs.append(sub_dir/task_name)
    print("Number of tasks to run in {}: {}".format(workdir, len(task_dirs)))
    num_parallel_processes = min(args.n_cpus, len(task_dirs))
    with ProcessPoolExecutor(max_workers=num_parallel_processes) as executor:
        results = list(executor.map(run_subprocess, task_dirs))
    for (stdout, stderr), task_dir in zip(results, task_dirs):
        with open(task_dir/"gcmc.out", "w") as f:
            f.write(stdout.decode("utf-8"))
        with open(task_dir/"gcmc.err", "w") as f:
            f.write(stderr.decode("utf-8"))

    