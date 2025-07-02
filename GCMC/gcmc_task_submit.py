'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 14:28:04
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from GCMC.utils import run_simulation
import argparse

from GCMC.gcmc_template import task_slurm_template
import time 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GCMC pre-task run')
    parser.add_argument('--n_cpus', type=int, default=300, help='Number of CPUs to use', required=False)
    parser.add_argument('--max_press', type=int, default=1000000, help='Maximum pressure to run', required=False)
    parser.add_argument('--min_press', type=int, default=0, help='Minimum pressure to run', required=False)
    args = parser.parse_args()

    prefix_id = 1
    root_dir = Path(os.path.abspath(__file__)).parent.parent
    print(root_dir)
    while True:
        prefix = "ddmof_batch{}".format(prefix_id)
        workdir = root_dir/("data/MOF_diversity/mc_data/{}".format(prefix))
        print("workdir: {}".format(workdir))
        if not workdir.exists():
            break
        script_name = "slurm_task_{}_{}.sh".format(args.min_press, args.max_press)
        if str(workdir/script_name) in [
            "/home/zhangsd/repos/MOF-MTHNN/data/MOF_diversity/mc_data/ddmof_batch43/slurm_task_100000_150000.sh",
            "/home/zhangsd/repos/MOF-MTHNN/data/MOF_diversity/mc_data/ddmof_batch51/slurm_task_100000_150000.sh",
            "/home/zhangsd/repos/MOF-MTHNN/data/MOF_diversity/mc_data/ddmof_batch61/slurm_task_100000_150000.sh",
            "/home/zhangsd/repos/MOF-MTHNN/data/MOF_diversity/mc_data/ddmof_batch78/slurm_task_300000_500000.sh",
            "/home/zhangsd/repos/MOF-MTHNN/data/MOF_diversity/mc_data/ddmof_batch64/slurm_task_300000_500000.sh"
        ]:
            prefix_id += 1
            continue

        with open(workdir/script_name, "w") as f:
            f.write(task_slurm_template.replace("PREFIX_ID", str(prefix_id)).\
                    replace("NUM_TASKS", str(args.n_cpus)).replace("MAX_P", str(args.max_press)).\
                    replace("MIN_P", str(args.min_press)))
        process = run_simulation(workdir, executor="sbatch", script_name=script_name)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print("Error: {}".format(stderr.decode()))
        else:
            print("Task job submitted for {} is done: ({})".format(prefix, stdout.decode()))
        prefix_id += 1
        time.sleep(1)
    