'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 16:36:48
'''

import os

abspath = os.path.abspath(__file__)
rootdir = os.path.dirname(os.path.dirname(abspath))

mc_default_params = {
    "NumberOfAdsorptionCycles": 5000,      # Number of adsorption MC cycles
    "NumberOfAdsorptionInitCycles": 5000,  # Number of adsorption MC cycles for initialization
    "NumberOfAdsorptionEqCycles": None,  # Number of adsorption MC cycles for equilibration
    "NumberOfHeliumVFCycles": 5000,  # Number of MC cycles for helium void fraction calculation
    "NumberOfQstInitCycles": 5000,  # Number of Qst MC cycles for initialization
    "NumberOfQstCycles": 5000,  # Number of Qst MC cycles for calculation
    "PrintEvery": 100,            # Print results every N cycles    
    "RestartFile": "no",          # Restart from previous simulation
    "ContinueAfterCrash": "no",   # Continue after crash
    "WriteBinaryRestartFileEvery": 1000,  # Write binary restart file every N cycles
    "RemoveAtomNumberCodeFromLabel": "no",  # Remove atom number code from label in input cif file

    "Forcefield": "UFF",          # Forcefield to use        
    "UseChargesFromCIFFile": "yes",  # Use charges from CIF file
    "ChargeMethod": "Ewald",      # Charge method to use
    "CutOff": 12.8,               # Cut-off distance for Ewald summation
    "EwaldPrecision": 1e-6,       # Precision for Ewald summation
    "TimeStep": 0.0005,           # Time step for MC simulation
    "CutOffRule": "truncated",    # Cut-off rule for non-bonded interactions: "truncated", "shifted"
    "TailCorrection": "no",       # Use tail correction for non-bonded interactions

    "ExternalTemperature": 298.0,  # External temperature
    "ExternalPressure": [100000.0,],  # External pressure
    "PressureMode": "parallel",   # Pressure mode: "parallel", "sequential"

    "AllComponents": ["CO2", "N2"],  # Components to simulate
    "MoleculeDefinition": "TraPPE",  # Molecule definition
    "TranslationProbability": 0.5,  # Probability of translation
    "ReinsertionProbability": 0.5,  # Probability of reinsertion
    "RotationProbability": 0.5,  # Probability of rotation    
    "SwapProbability": 1.0,  # Probability of swap
    "CreateNumberOfMolecules": 0,  # Number of molecules to create

    "MolFractions": [0.15, 0.85],  # Mole fractions of components
    "IdentityChangesList": [0, 1],  # List of identity changes
    "NumberOfIdentityChanges": 2,  # Number of identity changes

    "QstMethods": ["fluctuate"],  # Qst methods to use: "widom", "ideal", "fluctuate"
    "ChargeEqMethod": None,  # Charge equilibration method to use: None, "EQeq", "Qeq"
    "CalHeliumVF": True,  # Calculate helium void fraction

    "RASPA_DIR": "/opt/share/RASPA/simulations",  # Directory with RASPA installation
    "DefFileDir": os.path.join(rootdir, "GCMC/FF"),  # Directory with def files

    "TaskPrefix": "GCMC",  # Prefix for workdir name
    "TaskExecutor": "sbatch", # Task executor: "sbatch", "bash", or, "nohup"

}

pure_component_template = """
SimulationType                MonteCarlo
NumberOfCycles                10000
NumberOfInitializationCycles  10000
PrintEvery                    100
RestartFile                   no
ContinueAfterCrash            no
WriteBinaryRestartFileEvery   1000

Forcefield                    UFF
UseChargesFromCIFFile         yes
RemoveAtomNumberCodeFromLabel no

ChargeMethod                  Ewald
CutOff                        12.8
EwaldPrecision                1e-6  
TimeStep                      0.0005

Framework 0
FrameworkName MIL-47
UnitCells 1 1 1
HeliumVoidFraction 0.61
ExternalTemperature 298.0
ExternalPressure 100000.0

Component 0 MoleculeName             CO2
            MoleculeDefinition       TraPPE
            TranslationProbability   0.5
            ReinsertionProbability   0.5
            RotationProbability      0.5
            SwapProbability          1.0
            CreateNumberOfMolecules  0
""".strip()

binary_component_template = """
SimulationType                MonteCarlo
NumberOfCycles                5000
NumberOfEquilibrationCycles   0
NumberOfInitializationCycles  5000
PrintEvery                    1000
RestartFile                   no
ContinueAfterCrash            no
WriteBinaryRestartFileEvery   1000

Forcefield                    UFF
UseChargesFromCIFFile         yes
RemoveAtomNumberCodeFromLabel no

ChargeMethod                  Ewald
CutOff                        12.8
EwaldPrecision                1e-6  
TimeStep                      0.0005

Framework 0
FrameworkName MIL-47
UnitCells 1 1 1
HeliumVoidFraction 0.61
ExternalTemperature 313.0
ExternalPressure 100000.0

Component 0 MoleculeName               CO2
            MoleculeDefinition         TraPPE
            MolFraction                0.15
            TranslationProbability     0.5
            RotationProbability        0.5
            ReinsertionProbability     0.5
            IdentityChangeProbability  1.0
              NumberOfIdentityChanges  2
              IdentityChangesList      0 1
            SwapProbability            1.0
            CreateNumberOfMolecules    0

Component 1 MoleculeName               N2
            MoleculeDefinition         TraPPE
            MolFraction                0.85
            TranslationProbability     0.5
            RotationProbability        0.5
            ReinsertionProbability     0.5
            IdentityChangeProbability  1.0
              NumberOfIdentityChanges  2
              IdentityChangesList      0 1
            SwapProbability            1.0
            CreateNumberOfMolecules    0

""".strip()

qst_widom_template = """
SimulationType                MonteCarlo
NumberOfCycles                5000
NumberOfInitializationCycles  1000
PrintEvery                    1000
RestartFile                   no

Forcefield                    UFF
UseChargesFromCIFFile         yes
RemoveAtomNumberCodeFromLabel no

ChargeMethod                  Ewald
CutOff                        12.8
EwaldPrecision                1e-6  
TimeStep                      0.0005

Framework 0
FrameworkName MIL-47
UnitCells 4 2 2
HeliumVoidFraction 0.61
ExternalTemperature 313.0
ExternalPressure 0.0

Component 0 MoleculeName             N2
            MoleculeDefinition       ExampleDefinitions
            WidomProbability         1.0
            CBMCProbability          1.0
            CreateNumberOfMolecules  0
""".strip()

qst_inﬁnite_dilution_template = """
SimulationType                MonteCarlo
NumberOfCycles                10000
NumberOfInitializationCycles  100
PrintEvery                    1000
RestartFile                   no

Forcefield                    UFF
UseChargesFromCIFFile         yes
CutOff                        12.8
RemoveAtomNumberCodeFromLabel no

ChargeMethod                  Ewald
CutOff                        12.8
EwaldPrecision                1e-6  
TimeStep                      0.0005

Framework 0
FrameworkName AHINIP_clean
UnitCells 3 2 2
HeliumVoidFraction 0.183944
ExternalTemperature 313.0
ExternalPressure 0.0

Component 0 MoleculeName             CO2
            MoleculeDefinition       TraPPE
            TranslationProbability   0.5
            RotationProbability      0.5
            ReinsertionProbability   0.5
            CreateNumberOfMolecules  1
""".strip()


helium_vf_template = """
SimulationType                MonteCarlo
NumberOfCycles                10000
PrintEvery                    1000
PrintPropertiesEvery          1000

Forcefield                    ExampleMOFsForceField
RemoveAtomNumberCodeFromLabel no

Framework 0
FrameworkName MIL-47
UnitCells 4 2 2
ExternalTemperature 313.0
#HeliumVoidFraction 0.61

Component 0 MoleculeName             helium
            MoleculeDefinition       ExampleDefinitions
            WidomProbability         1.0
            CreateNumberOfMolecules  0
""".strip()

charge_eq_template = """
SimulationType                          MonteCarlo
NumberOfCycles                          0
NumberOfInitializationCycles            0
PrintEvery                              100
RestartFile                             no


Forcefield                              UFF
CutOff                                  12.8
RemoveAtomNumberCodeFromLabel           no


ChargeFromChargeEquilibration           yes
ChargeEquilibrationPeriodic             yes
ChargeEquilibrationEwald                yes
SymmetrizeFrameworkCharges              no


Framework 0
FrameworkName IRMOF-3
UnitCells 1 1 1
ExternalTemperature 298.0
ExternalPressure 0.0
""".strip()


slurm_template = """#!/bin/bash
#SBATCH --job-name=gcmc
#SBATCH --output=gcmc.out
#SBATCH --error=gcmc.err
#SBATCH --partition=C9654 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000M

export RASPA_DIR=/opt/share/RASPA/simulations
export DYLD_LIBRARY_PATH=${RASPA_DIR}/lib
export LD_LIBRARY_PATH=${RASPA_DIR}/lib

srun $RASPA_DIR/bin/simulate
"""

slurm_array_template = """#!/bin/bash
#SBATCH --job-name=gcmc
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=C9654
#SBATCH --nodes=1
#SBATCH --ntasks=NUM_TASKS
#SBATCH --cpus-per-task=1
#SBATCH --array=1-NUM_ARRAY_TASKS

export RASPA_DIR=/opt/share/RASPA/simulations
export DYLD_LIBRARY_PATH=${RASPA_DIR}/lib
export LD_LIBRARY_PATH=${RASPA_DIR}/lib

# 读取子文件夹列表到数组中
mapfile -t folders < task_folders.txt

# 获取当前任务的文件夹
folder=${folders[$SLURM_ARRAY_TASK_ID-1]}

# 进入子文件夹
cd $folder
echo "Current folder: $folder"

# 执行run命令
srun $RASPA_DIR/bin/simulate
"""

bash_template = """#!/bin/bash

export RASPA_DIR=/opt/share/RASPA/simulations
export DYLD_LIBRARY_PATH=${RASPA_DIR}/lib
export LD_LIBRARY_PATH=${RASPA_DIR}/lib

$RASPA_DIR/bin/simulate
"""

pre_task_slurm_template = """#!/bin/bash
#SBATCH --job-name=gcmc_batchPREFIX_ID
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks=NUM_TASKS
#SBATCH --nodes=1
export PATH=/opt/share/miniconda3/envs/lammps/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/lammps/lib/:$LD_LIBRARY_PATH

python -u gcmc_pre_task_run.py --prefix_id PREFIX_ID --n_cpus NUM_TASKS"""


task_slurm_template = """#!/bin/bash
#SBATCH --job-name=gcmc_batchPREFIX_ID
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=C9654 
#SBATCH --ntasks=NUM_TASKS
#SBATCH --nodelist=c[1,4-5]
#SBATCH --nodes=1
export PATH=/opt/share/miniconda3/envs/lammps/bin/:$PATH
export LD_LIBRARY_PATH=/opt/share/miniconda3/envs/lammps/lib/:$LD_LIBRARY_PATH

python -u gcmc_task_run.py --prefix_id PREFIX_ID --n_cpus NUM_TASKS --max_press MAX_P --min_press MIN_P"""