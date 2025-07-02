'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 14:27:33
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from GCMC.utils import run_simulation

from gcmc_template import pre_task_slurm_template


if __name__ == "__main__":

    prefix_id = 1
    n_cpus = 32
    root_dir = Path(os.path.abspath(__file__)).parent.parent
    print(root_dir)
    while True:
        prefix = "ddmof_batch{}".format(prefix_id)
        workdir = root_dir/("data/MOF_diversity/mc_data/{}".format(prefix))
        print("workdir: {}".format(workdir))
        if not workdir.exists():
            break
        with open(workdir/"slurm_pre_task.sh", "w") as f:
            f.write(pre_task_slurm_template.replace("PREFIX_ID", str(prefix_id)).replace("NUM_TASKS", str(n_cpus)))
        process = run_simulation(workdir, executor="sbatch", script_name="slurm_pre_task.sh")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print("Error: {}".format(stderr.decode()))
        else:
            print("Pre-task job submitted for {} is done: ({})".format(prefix, stdout.decode()))
        prefix_id += 1

        

    