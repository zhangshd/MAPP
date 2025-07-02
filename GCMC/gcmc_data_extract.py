'''
Author: zhangshd
Date: 2024-08-28 21:06:25
LastEditors: zhangshd
LastEditTime: 2024-09-10 14:26:54
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GCMC.utils import process_isotherm_results, process_heat_results
from pathlib import Path
import pandas as pd


root_dir = Path(os.path.abspath(__file__)).parent.parent
print(root_dir)
gases = ["CO2", "N2"]
n_jobs = 32
for prefix_id in range(1, 90):
    workdir = root_dir/("data/MOF_diversity/mc_data/ddmof_batch{}".format(prefix_id))
    print("-"*50)
    print("Work directory: ", workdir)
    result_dir = root_dir/"data/MOF_diversity/mc_data_tabular"
    result_dir.mkdir(exist_ok=True, parents=True)
    # df_isotherm = process_isotherm_results(workdir, gases, unit="mol/kg", verbose=0, n_jobs=n_jobs)
    # print("number of isotherm data points: ", len(df_isotherm))
    # if len(df_isotherm) > 0:
    #     df_isotherm.to_csv(result_dir/('00-isotherm-data-ddmof_batch{}.tsv'.format(prefix_id)), index=False, sep='\t')
    df_qst = process_heat_results(workdir, gases, verbose=0, n_jobs=n_jobs)
    print("number of Qst data points: ", len(df_qst))
    if len(df_qst) > 0:
        df_qst.to_csv(result_dir/('00-Qst-data-ddmof_batch{}.tsv'.format(prefix_id)), index=False, sep='\t')