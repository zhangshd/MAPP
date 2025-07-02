'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 14:26:38
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from GCMC.gcmc_process import create_tasks
import time
import random
from decimal import Decimal
import numpy as np

def main(prefix_id, sample_num, all_candidates_names=None):
    user_defined_params = {
        "NumberOfAdsorptionCycles": 5000,      # Number of adsorption MC cycles
        "NumberOfAdsorptionInitCycles": 5000,  # Number of adsorption MC cycles for initialization
        "NumberOfAdsorptionEqCycles": 0,  # Number of adsorption MC cycles for equilibration
        "NumberOfHeliumVFCycles": 5000,  # Number of MC cycles for helium void fraction calculation
        "NumberOfQstInitCycles": 5000,  # Number of Qst MC cycles for initialization
        "NumberOfQstCycles": 5000,  # Number of Qst MC cycles for calculation
        "PrintEvery": 100,            # Print results every N cycles    
        "RestartFile": "no",          # Restart from previous simulation
        "RemoveAtomNumberCodeFromLabel": "yes",  # Remove atom number code from label in input cif file

        "Forcefield": "UFF",          # Forcefield to use        
        "UseChargesFromCIFFile": "yes",  # Use charges from CIF file
        "ChargeMethod": "Ewald",      # Charge method to use
        "CutOff": 12.8,               # Cut-off distance for Ewald summation
        "EwaldPrecision": 1e-6,       # Precision for Ewald summation
        "TimeStep": 0.0005,           # Time step for MC simulation
        "CutOffRule": "shifted",    # Cut-off rule for non-bonded interactions: "truncated", "shifted"
        "TailCorrection": "yes",       # Use tail correction for non-bonded interactions

        "ExternalTemperature": 298.0,  # External temperature
        "ExternalPressure": [round(0.001*i*100000, 7) for i in [0.1, 1, 10, 100, 1000]], # External pressure
        "PressureMode": "parallel",   # Pressure mode: "parallel", "sequential"

        "AllComponents": ["CO2", "N2"],  # Components to simulate
        "MoleculeDefinition": "TraPPE",  # Molecule definition
        "TranslationProbability": 0.5,  # Probability of translation
        "ReinsertionProbability": 0.5,  # Probability of reinsertion
        "RotationProbability": 0.5,  # Probability of rotation    
        "SwapProbability": 1.0,  # Probability of swap
        "CreateNumberOfMolecules": 0,  # Number of molecules to create

        "MolFractions": [[0.15, 0.85], [0.1, 0.9]],  # Mole fractions of components
        "IdentityChangesList": [0, 1],  # List of identity changes
        "NumberOfIdentityChanges": 2,  # Number of identity changes

        "AdsorptionComponetTypes": ["pure", "mixture"],  # Adsorption component type: "pure", "mixture"
        "QstMethods": ["fluctuate"],  # Qst methods to use: "widom", "ideal", "fluctuate"
        "ChargeEqMethod": "Qeq",  # Charge equilibration method to use: None, "EQeq", "Qeq"
        "CalHeliumVF": True,  # Calculate helium void fraction

        "TaskPrefix": "ddmof_batch{}".format(prefix_id),  # Prefix for workdir name
        "TaskExecutor": "bash", # Task executor: "sbatch", "sbatch_array", "bash", or, "nohup"
        "RandomSeed": prefix_id,  # Random seed for reproducibility
    }


    root_dir = Path(os.path.abspath(__file__)).parent.parent
    print(root_dir)
    src_cif_dir = root_dir/"data/ddmof/mof_data/ddmof_cifs"
    workdir = root_dir/("data/ddmof/mc_data/{}".format(user_defined_params["TaskPrefix"], ))
    print("workdir: {}".format(workdir))


    random.seed(user_defined_params["RandomSeed"])
    np.random.seed(user_defined_params["RandomSeed"])

    if all_candidates_names is None:
        exclued_names = []
        for wd in workdir.parent.glob("*"):
            if wd.is_dir() and not wd.name.startswith(user_defined_params["TaskPrefix"]):
                for d in wd.glob("*"):
                    if d.is_dir():
                        exclued_names.append(d.name)
        print("Number of excluded names: ", len(exclued_names))
        all_candidates_names = []
        for file in os.listdir(src_cif_dir):
            name = file.replace(".cif", "")
            if name not in exclued_names:
                all_candidates_names.append(name)
        print("Number of candidates: ", len(all_candidates_names))
        mof_list = [file.replace(".cif", "") for file in np.random.choice(all_candidates_names, 
                                                                          min(sample_num, len(all_candidates_names)),  
                                                                          replace=False)]

        select_name_list = []
        for mof_name in mof_list:
            if (src_cif_dir/(mof_name + ".cif")).exists():
                select_name_list.append(mof_name)
        print("Number of selected MOFs: ", len(select_name_list))
        select_name_list.sort()

        assert len(select_name_list) + len(set(exclued_names)) == len(set(select_name_list + exclued_names))
    else:
        mof_list = [file.replace(".cif", "") for file in np.random.choice(all_candidates_names, 
                                                                          min(sample_num, len(all_candidates_names)), 
                                                                          replace=False)]
        select_name_list = []
        for mof_name in mof_list:
            if (src_cif_dir/(mof_name + ".cif")).exists():
                select_name_list.append(mof_name)
        print("Number of selected MOFs: ", len(select_name_list))
        select_name_list.sort()

    # randomly select pressure points and mole fractions
    press_points = [0.001*i for i in [1, 10, 50, 100, 200, 400, 600, 800, 1000]] + list(range(2, 5, 2))
    press_points = [abs(val + np.random.normal(scale=0.5*val)) for val in press_points]
    press_points = [val*100000 for val in press_points]
    press_points = [Decimal(v).quantize(Decimal('1.0')) for v in press_points]
    press_points = [float(v) for v in press_points]

    mol_fracs = [Decimal(0.01*v).quantize(Decimal('1.00')) for v in np.random.randint(10, 90, size=3)]
    mol_fracs = [[float(v), 1-float(v)] for v in mol_fracs]

    user_defined_params["ExternalPressure"] = press_points
    user_defined_params["MolFractions"] = mol_fracs
    print("User defined params: \n", user_defined_params)

    ## create tasks
    task_dirs, task_tuples = create_tasks(workdir, 
                                      select_name_list, 
                                      src_cif_dir, 
                                      user_defined_params, 
                                      overwrite=True,
                                      verbose=1
                                      )
    all_candidates_names = [n for n in all_candidates_names if n not in select_name_list]
    return all_candidates_names
    

if __name__ == "__main__":


    t0 = time.time()
    i = 32
    sample_num = 300
    all_candidates_names = main(i, sample_num, all_candidates_names=None)
    while len(all_candidates_names) > 0:
        t1 = time.time()
        print("/"*50)
        print("Number of remaining candidates: ", len(all_candidates_names))
        print("/"*50)
        i += 1
        all_candidates_names = main(i, 300, all_candidates_names)
        print("Time used in this iteration: ", time.strftime('%H:%M:%S', time.gmtime(time.time()-t1)))
        print("Time used totally: ", time.strftime('%H:%M:%S', time.gmtime(time.time()-t0)))