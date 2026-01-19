'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-11-29 22:49:53
'''
import subprocess
from pathlib import Path
import os
import time
#SBATCH --nodelist=c[2-3]
job_templet = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --partition=C9654
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --time=120:00:00
export PATH=/opt/share/miniconda3/envs/mofnn/bin:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/mofnn/lib:$LD_LIBRARY_PATH

srun python -u {py_executor} --progress_bar --task_cfg {task_config} --model_cfg {model_config}
""".strip()

def run_slurm_job(work_dir, executor="sbatch", script_name="run"):
    work_dir = Path(work_dir)
    # Create a script to run the job
    process = subprocess.Popen(
        f"{executor} {work_dir/script_name}",
        # [executor, str(work_dir/'run'), "&"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(work_dir)
    )
    return process

if __name__ == '__main__':
    work_dir = Path("./")

    task_configs = [
        # "ads_s_qst_co2_n2",
        # "ads_qst_co2_n2",
        # "ads_co2_n2",
        # "ads_s_co2_n2",
        "ads_symlog_co2_n2",
        # "ads_co2",
        # "ads_n2",
        # "qst_co2",
        # "qst_n2",
        
                     ]
    model_configs = [
        # "att_cgcnn",
        # "cgcnn",
        "cgcnn_langmuir",
        # "cgcnn_raw",
        # "fcnn",
        # "att_fcnn",
        # "cgcnn_uni_atom"
    ]
    script_name = "run_slurm.sh"
    # py_executor = "hyperopt.py"
    py_executor = "main.py"
    model_conf = {
                'batch_size': 32,
                # 'limit_train_batches': 0.8,
                'max_epochs': 50, 
                # 'max_graph_len': 200,
                'atom_fea_len': 64,
                'extra_fea_len': 128,
                'h_fea_len': 128,
                'n_conv': 3,
                'n_h': 1,
                'dropout_prob': 0.5,
                'use_cell_params': False,
                'atom_layer_norm': False,
                'loss_aggregation': "sum",   # fixed_weight_sum, dwa, sum, sample_weight_sum, trainable_weight_sum
                # 'dl_sampler': 'random',
                # 'task_att_type': 'self',
                'lr': 0.0001,
                'lr_mult': 1,
                # 'group_lr': True,
                'optim_config': "fine",  # fine or coarse
                'auto_lr_bs_find': False, 
                'patience': 10,
                # 'att_pooling': False,
                'task_norm': True,
                # 'reconstruct': False,
                'log_dir': "logs",
                'devices': 2,
                # 'optuna_name': "optuna",
                }
    
    for task_config in task_configs:
        for model_config in model_configs:
            job_name = f"{task_config.replace('_config', '')}_{model_config.replace('_config', '')}"
            if py_executor == "hyperopt.py":
                job_name = "opt_" + job_name
                # job_templet_ = job_templet + " --pruning"
                job_templet_ = job_templet
            else:
                job_templet_ = job_templet
            job_script = job_templet_.format(job_name=job_name, 
                                            task_config=task_config, 
                                            model_config=model_config,
                                            py_executor=py_executor,
                                            n_gpus=model_conf["devices"]
                                            )
            
            for key, value in model_conf.items():
                if isinstance(value, bool):
                    if value:
                        job_script += f" --{key}"
                    continue
                job_script += f" --{key} {value}"
            with open(work_dir/script_name, "w") as f:
                f.write(job_script)
            process = run_slurm_job(work_dir, executor="sbatch", script_name=script_name)
            ## get the output of the job
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.decode().strip())
            print(f"Submitted job {job_name} with PID {process.pid}")
            time.sleep(1)
