'''
Author: zhangshd
Date: 2024-09-09 21:25:02
LastEditors: zhangshd
LastEditTime: 2024-10-30 16:50:47
'''

from pathlib import Path
import shutil
import re
import asyncio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GCMC.utils import chxmod, calculate_repetitions
from GCMC.utils import get_void_fraction, set_simulation_params
from GCMC.utils import set_ff_params, set_raspa_dir
from GCMC.utils import run_simulation, run_simulation_async
from GCMC.molecule_utils import cal_atom_charge
from GCMC.gcmc_template import pure_component_template, binary_component_template,  \
    helium_vf_template, charge_eq_template, qst_infinite_dilution_template, \
    qst_widom_template, slurm_template, bash_template, mc_default_params, slurm_array_template
import datetime
import time
from typing import List, Union
import json


def parse_task(user_defined_params):

    task_names = []
    if "pure" in user_defined_params["AdsorptionComponetTypes"] or "fluctuate" in user_defined_params["QstMethods"]:
        task_names.extend(["Adsorption_pure_" + g for g in user_defined_params["AllComponents"]])
    if "mixture" in user_defined_params["AdsorptionComponetTypes"]:
        if isinstance(user_defined_params["MolFractions"], (list, tuple)) and isinstance(user_defined_params["MolFractions"][0], (int, float)):
            user_defined_params["MolFractions"] = [user_defined_params["MolFractions"]]
        for mol_fracs in user_defined_params["MolFractions"]:
            task_names.extend(["Adsorption_" + \
                            "_".join(user_defined_params["AllComponents"]) + "_" + \
                            "_".join(["{:.3f}".format(f) for f in mol_fracs])])

    ## parallelly run adsorption tasks on each pressure point independently. 
    if user_defined_params["PressureMode"] == "parallel":
        task_tuples = []
        for task in task_names:
            for press in user_defined_params["ExternalPressure"]:
                task_tuples.append([task + "_{:.3f}bar".format(press/100000), press])
    ## Sequentially run an adsorption task on mutilple pressure points from low to high.
    else:
        task_tuples = [[task, user_defined_params["ExternalPressure"]] for task in task_names]

    if "widom" in user_defined_params["QstMethods"]:
        task_tuples.extend([["Qst_widom_" + g, 0] for g in user_defined_params["AllComponents"]])
    if "ideal" in user_defined_params["QstMethods"]:
        task_tuples.extend([["Qst_ideal_" + g, 0] for g in user_defined_params["AllComponents"]])

    return task_tuples

def create_tasks(workdir: Union[str, Path], 
                mof_names: List[str], 
                src_cif_dir: Union[str, Path], 
                simu_params: dict=None,
                overwrite: bool=False,
                verbose: int=0,
                ):

    mc_default_params.update(simu_params)

    task_tuples = parse_task(mc_default_params)
    mc_default_params["TaskTuples"] = [tuple(t) for t in task_tuples]


    auxiliary_param_keys = [
        "ExternalPressure",
        "AdsorptionComponetTypes",
        "QstMethods",
        "TaskPrefix",
        "TaskExecutor",
        "TaskTuples"
    ]
    workdir = Path(workdir)
    mc_params_from_file = mc_default_params.copy()
    
    if workdir.exists() and overwrite:
        shutil.rmtree(str(workdir))
    elif workdir.exists():
        with open(workdir/"00_gcmc_params.json", "r") as f:
            mc_params_from_file = json.load(f)
        for key in mc_default_params.keys():
            if key in auxiliary_param_keys:
                continue
            assert mc_params_from_file[key] == mc_default_params[key], \
            """Found unmatched parameter: {}: {} vs. {}. 
Please check the consistency of input parameters or set overwrite=True to overwrite the existing files.""".format(key, 
                                                                                                                  mc_params_from_file[key], 
                                                                                                                  mc_default_params[key])
        mc_params_from_file["TaskTuples"] = [tuple(t) for t in mc_params_from_file["TaskTuples"]]
        for key in auxiliary_param_keys:
            if not isinstance(mc_params_from_file[key], list):
                continue
            # print(mc_params_from_file[key], mc_default_params[key])
            mc_params_from_file[key] = sorted(set(mc_params_from_file[key] + mc_default_params[key]), 
                                              key=lambda x: x[0] if isinstance(x, tuple) else x)

    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir/"00_gcmc_params.json", "w") as f:
        json.dump(mc_params_from_file, f, indent=4)

    ## create sub-directories for each MOF and task
    task_dirs = []
    for mof_name in mof_names:
        
        file = mof_name + ".cif"
        sub_dir = workdir/(mof_name)
        sub_dir.mkdir(exist_ok=True, parents=True)
        if verbose > 0:
            print("sub dir: {}".format(sub_dir))

        ## create sub-directories for pre-tasks
        if mc_default_params["CalHeliumVF"]:
            pre_task_tuples = [["_helium_void_fraction", None]]
        else:
            pre_task_tuples = []

        ## copy cif file and modify atom charge if necessary
        if mc_default_params["ChargeEqMethod"] == "EQeq":
            cif_with_charge = cal_atom_charge(in_cif=src_cif_dir/file, overwite=True, sanitize=False)
        elif mc_default_params["ChargeEqMethod"] == "Qeq":
            pre_task_tuples.append(["_charge_eq", 0])
            cif_with_charge = src_cif_dir/file
        else:
            cif_with_charge = src_cif_dir/file
        
        for task_name, press in pre_task_tuples + task_tuples:
            task_dir = sub_dir/task_name
            if task_dir.exists() and not overwrite:
                continue
            task_dir.mkdir(exist_ok=True, parents=True)
            if verbose > 1:
                print("="*50)
                print("task dir: {}".format(task_dir))
            
            shutil.copy(cif_with_charge, str(task_dir/file))   ## copy cif file

            ## create run script
            if mc_default_params["TaskExecutor"] == "sbatch":
                run_content = slurm_template.replace("--job-name=gcmc", 
                                                        "--job-name=gcmc_{}".format(mc_default_params["TaskPrefix"]))
                with open(task_dir/"run", "w") as f:
                    f.write(run_content)
            else:
                run_content = bash_template
                with open(task_dir/"run", "w") as f:
                    f.write(run_content)
            
            ## copy force field def files and molecule def files
            DefFileDir = Path(mc_default_params["DefFileDir"])
            for def_file in (DefFileDir/mc_default_params["Forcefield"]).glob("*.def"):
                shutil.copy(str(def_file), str(task_dir))       ## UFF def files
            for def_file in (DefFileDir/mc_default_params["MoleculeDefinition"]).glob("*.def"):
                if def_file.name.split(".")[0].upper() not in task_name.upper().split("_"):
                    continue
                shutil.copy(str(def_file), str(task_dir))       ## molecule def files
            ## set force_field_mixing_rules.def
            set_ff_params(task_dir/"force_field_mixing_rules.def", 
                          cutoff_rule=mc_default_params["CutOffRule"],
                          tail_correction=mc_default_params["TailCorrection"],
                          )
            ## set run script parameters
            set_raspa_dir(task_dir/"run", raspa_dir=mc_default_params["RASPA_DIR"])
            chxmod(task_dir/"run")
                
            ## get repetitions of unit cell according to cutoff_radius of force field
            repetitions = calculate_repetitions(task_dir/file, cutoff_radius=mc_default_params["CutOff"])
            repetitions = " ".join([str(r) for r in repetitions])
            components = re.findall("|".join(mc_default_params["AllComponents"]), task_name)
            

            ## set simulation parameters for charge equilibration
            if task_name == "_charge_eq":
                if verbose > 1:
                    print("setting charge equilibration parameters")
                with open(task_dir/"simulation.input", "w") as f:
                    f.write(charge_eq_template)
                set_simulation_params(task_dir, temp=mc_default_params["ExternalTemperature"], 
                                      press=0, ff=mc_default_params["Forcefield"], 
                                      unitcells=None, framework_name=sub_dir.name,
                                      remove_atom_number_code_from_label=mc_default_params["RemoveAtomNumberCodeFromLabel"],
                                        )
            ## set simulation parameters for void fraction calculation
            elif task_name == "_helium_void_fraction":
                if verbose > 1:
                    print("setting charge helium void fraction")
                with open(task_dir/"simulation.input", "w") as f:
                    f.write(helium_vf_template)
                set_simulation_params(task_dir, temp=mc_default_params["ExternalTemperature"], 
                                      press=None, ff=mc_default_params["Forcefield"], 
                                      unitcells=repetitions,
                                      moldef=mc_default_params["MoleculeDefinition"], 
                                      init_cycles=None,
                                      cycles=mc_default_params["NumberOfHeliumVFCycles"], 
                                      framework_name=sub_dir.name,
                                      components="helium", 
                                      remove_atom_number_code_from_label=mc_default_params["RemoveAtomNumberCodeFromLabel"],
                                      verbose=verbose,
                                    )
            ## set simulation parameters for Qst calculation using Widom insertion
            elif "Qst_widom" in task_name:
                if verbose > 1:
                    print("setting Qst calculation using Widom insertion")
                with open(task_dir/"simulation.input", "w") as f:
                    f.write(qst_widom_template)
                set_simulation_params(task_dir, temp=mc_default_params["ExternalTemperature"], 
                                      press=press, ff=mc_default_params["Forcefield"], 
                                      unitcells=repetitions,
                                      moldef=mc_default_params["MoleculeDefinition"], 
                                      init_cycles=mc_default_params["NumberOfQstInitCycles"], 
                                      cycles=mc_default_params["NumberOfQstCycles"], 
                                      framework_name=sub_dir.name,
                                      components=components,
                                      remove_atom_number_code_from_label=mc_default_params["RemoveAtomNumberCodeFromLabel"],
                                        )
            ## set simulation parameters for Qst calculation using infinite dilution condition
            elif "Qst_ideal" in task_name:
                if verbose > 1:
                    print("setting Qst calculation using infinite dilution condition")
                with open(task_dir/"simulation.input", "w") as f:
                    f.write(qst_inﬁnite_dilution_template)
                set_simulation_params(task_dir, temp=mc_default_params["ExternalTemperature"], 
                                      press=press, ff=mc_default_params["Forcefield"], 
                                      unitcells=repetitions,
                                      moldef=mc_default_params["MoleculeDefinition"], 
                                      init_cycles=mc_default_params["NumberOfQstInitCycles"], 
                                      cycles=mc_default_params["NumberOfQstCycles"], 
                                      framework_name=sub_dir.name,
                                      components=components,
                                      remove_atom_number_code_from_label=mc_default_params["RemoveAtomNumberCodeFromLabel"],
                                        )
            ## set simulation parameters for pure component adsorption
            elif len(components) == 1:
                if verbose > 1:
                    print("setting pure component adsorption")
                with open(task_dir/"simulation.input", "w") as f:
                    f.write(pure_component_template)
                set_simulation_params(task_dir, temp=mc_default_params["ExternalTemperature"], 
                                      press=press, ff=mc_default_params["Forcefield"], 
                                      unitcells=repetitions,
                                      moldef=mc_default_params["MoleculeDefinition"], 
                                      init_cycles=mc_default_params["NumberOfAdsorptionInitCycles"],
                                      cycles=mc_default_params["NumberOfAdsorptionCycles"], 
                                      framework_name=sub_dir.name,
                                      components=components,
                                      remove_atom_number_code_from_label=mc_default_params["RemoveAtomNumberCodeFromLabel"],
                                        )
            ## set simulation parameters for binary mixture adsorption
            elif len(components) == 2:
                if verbose > 1:
                    print("setting binary mixture adsorption")
                mol_fracs = re.findall(r"{}_(\d+\.\d+)_(\d+\.\d+)".format("_".join(components)), task_name)[0]
                mol_fracs = [float(f) for f in mol_fracs]

                with open(task_dir/"simulation.input", "w") as f:
                    f.write(binary_component_template)
                set_simulation_params(task_dir, temp=mc_default_params["ExternalTemperature"], 
                                      press=press, ff=mc_default_params["Forcefield"], 
                                        unitcells=repetitions,
                                        moldef=mc_default_params["MoleculeDefinition"], 
                                        init_cycles=mc_default_params["NumberOfAdsorptionInitCycles"],
                                        cycles=mc_default_params["NumberOfAdsorptionCycles"], 
                                        framework_name=sub_dir.name,
                                        components=components,
                                        molfractions=mol_fracs,
                                        remove_atom_number_code_from_label=mc_default_params["RemoveAtomNumberCodeFromLabel"],
                                        )
            task_dirs.append(task_dir)

    if mc_default_params["TaskExecutor"] != "sbatch_array":
        return task_dirs, task_tuples
    
    slurm_log_dir = workdir/"slurm_logs"
    slurm_log_dir.mkdir(exist_ok=True, parents=True)
    vf_task_dirs = [d for d in task_dirs if "_helium_void_fraction" in d.name]
    charge_eq_task_dirs = [d for d in task_dirs if "_charge_eq" in d.name]
    mc_task_dirs = [d for d in task_dirs if not d.name.startswith("_")]
    task_dir_folder_files = [
        "vf_task_dirs.txt",
        "charge_eq_task_dirs.txt",
        "mc_task_dirs.txt",
    ]
    task_run_files = [
        "run_vf_task",
        "run_charge_eq_task",
        "run_mc_task",
    ]
    for task_list, task_list_file, task_run_file in zip([vf_task_dirs, charge_eq_task_dirs, mc_task_dirs], 
                                                      task_dir_folder_files, 
                                                      task_run_files):
        with open(workdir/task_list_file, "w") as f:
            f.write("\n".join([str(d) for d in task_list]))
        run_content = slurm_array_template.replace("--job-name=gcmc", 
                                                    "--job-name=gcmc_{}".format(mc_default_params["TaskPrefix"])).\
                            replace("task_folders.txt", task_list_file).replace("NUM_TASKS", str(1)).\
                            replace("NUM_ARRAY_TASKS", str(len(task_list))).\
                            replace("%x_%A_%a", str(slurm_log_dir/"%x_%A_%a"))
        with open(workdir/task_run_file, "w") as f:
            f.write(run_content)
        
    return task_dirs, task_tuples


def run_and_check_pre_tasks(workdir: Union[str, Path],
                            task_tuples: List[tuple]=None,
                            interval: int = 60,
                            num_interval: int = 5,
                            executor="sbatch",
                            check_only: bool = False,
                            verbose: int = 0,
                            ):
    workdir = Path(workdir)

    if task_tuples is None:
        with open(workdir/"00_gcmc_params.json", "r") as f:
                mc_params_from_file = json.load(f)
        task_tuples = mc_params_from_file["TaskTuples"]

    if not check_only and executor != "sbatch_array":
        ## run charge equilibration simulations if necessary
        for sub_dir in workdir.glob("*"):
            if not sub_dir.is_dir():
                continue
            ## run charge equilibration simulations if necessary
            charge_dir = sub_dir/"_charge_eq"
            if charge_dir.exists() and not (charge_dir/"Movies/System_0/").exists():
                print("Sumbitting charge equilibration simulations...")
                process = run_simulation(charge_dir, executor=executor)
                stdout, stderr = process.communicate()
    elif not check_only and executor == "sbatch_array" and (workdir/"run_charge_eq_task").exists():
        ## run charge equilibration simulations if necessary
        print("Sumbitting charge equilibration simulations...")
        process = run_simulation(workdir, executor="sbatch", script_name="run_charge_eq_task")
        stdout, stderr = process.communicate()
        # print(stderr.decode())

    print("Checking if all charge equilibration simulations are finished...")
    time_count = 0
    finished_dirs = set()
    while True:
        flag = []
        unfinished_dirs = []
        for sub_dir in sorted(workdir.glob("*")):
            if not sub_dir.is_dir():
                continue
            elif sub_dir.name in finished_dirs:
                flag.append(True)
                continue
            charge_dir = sub_dir/"_charge_eq"
            if charge_dir.exists():
                if not (charge_dir/"Movies/System_0/").exists():
                    flag.append(False)
                    unfinished_dirs.append((sub_dir, "unfinished"))
                    continue
                else:
                    flag.append(True)
                    finished_dirs.add(sub_dir.name)
        print("{}/{}".format(sum(flag), len(flag)))
        time_count += interval
        if all(flag):
            print("charge equilibration simulations all done!")
            break
        elif time_count >= interval*num_interval:
            print("charge equilibration simulations not finished after {} minites, force stop!".format(time_count/60))
            print("number of unfinished simulations: {}".format(len(unfinished_dirs)))
            unfinished_report = workdir/"unfinished_dirs_charge_eq_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            with open(unfinished_report, "w") as f:
                for d, reason in unfinished_dirs:
                    f.write("{}: {}\n".format(d, reason))
            print("information of unfinished simulations saved in {}".format(unfinished_report))
            break
        time.sleep(interval)
    unfinished_dirs_charge_eq = unfinished_dirs

    for sub_dir in workdir.glob("*"):
        if not sub_dir.is_dir():
            continue
        ## copy charged cif file to each task directory
        for task_name, press in task_tuples:
            task_dir = sub_dir/task_name
            charged_cifs = list((sub_dir/"_charge_eq/Movies/System_0/").glob("Framework_*_final_*P1.cif"))
            if len(charged_cifs) > 0:
                shutil.copy(charged_cifs[0], str(task_dir/(sub_dir.name + ".cif")))   ## cif file
                if verbose > 1:
                    print("{} >> {}".format(charged_cifs[0], task_dir/(sub_dir.name + ".cif")))
                
        if not check_only and executor != "sbatch_array":
            ## run helium void fraction simulations if necessary
            vf_dir = sub_dir/"_helium_void_fraction"
            if vf_dir.exists() and not (vf_dir/"Output/System_0/").exists():
                charged_cifs = list((sub_dir/"_charge_eq/Movies/System_0/").glob("Framework_*_final_*P1.cif"))
                if len(charged_cifs) > 0:
                    shutil.copy(charged_cifs[0], str(vf_dir/(sub_dir.name + ".cif")))   ## cif file
                    if verbose > 1:
                        print("{} >> {}".format(charged_cifs[0], vf_dir/(sub_dir.name + ".cif")))
                print("Sumbitting helium void fraction simulations...")
                process = run_simulation(vf_dir, executor=executor)
                stdout, stderr = process.communicate()
    if not check_only and executor == "sbatch_array" and (workdir/"run_vf_task").exists():
        ## run charge equilibration simulations if necessary
        print("Sumbitting helium void fraction simulations...")
        process = run_simulation(workdir, executor="sbatch", script_name="run_vf_task")
        stdout, stderr = process.communicate()
        # print(stderr.decode())

    print("Checking if all helium void fraction simulations are finished...")
    time_count = 0
    finished_dirs = []
    while True:
        flag = []
        unfinished_dirs = []
        for sub_dir in sorted(workdir.glob("*")):
            if not sub_dir.is_dir():
                continue
            elif sub_dir.name in finished_dirs:
                flag.append(True)
                continue
            vf_dir = sub_dir/"_helium_void_fraction"
            if vf_dir.exists():
                data_files = list((vf_dir/"Output/System_0/").glob("*.data"))
                if len(data_files) == 0:
                    flag.append(False)
                    unfinished_dirs.append((sub_dir, "No data file found"))
                    continue
                with open(data_files[0], "r") as f:
                    content = f.read().strip()
                if "The end time was" in content:
                    flag.append(True)
                    finished_dirs.append(sub_dir.name)
                else:
                    flag.append(False)
                    unfinished_dirs.append((sub_dir, "unfinished"))

        print("{}/{}".format(sum(flag), len(flag)))
        time_count += interval
        if all(flag):
            print("helium void fraction simulations all done!")
            break
        elif time_count >= interval*num_interval:
            print("helium void fraction simulations not finished after {} minites, force stop!".format(time_count/60))
            print("number of unfinished simulations: {}".format(len(unfinished_dirs)))
            unfinished_report = workdir/"unfinished_dirs_void_fraction_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            with open(unfinished_report, "w") as f:
                for d, reason in unfinished_dirs:
                    f.write("{}: {}\n".format(d, reason))
            print("information of unfinished simulations saved in {}".format(unfinished_report))
            break
        time.sleep(interval)
    unfinished_dirs_void_fraction = unfinished_dirs

    return unfinished_dirs_charge_eq, unfinished_dirs_void_fraction

async def run_and_check_pre_tasks_async(workdir: Union[str, Path],
                            task_tuples: List[tuple],
                            interval: int = 20,
                            ):
    workdir = Path(workdir)

    for sub_dir in workdir.glob("*"):
        if not sub_dir.is_dir():
            continue
        ## run charge equilibration simulations if necessary
        charge_dir = sub_dir/"_charge_eq"
        if charge_dir.exists():
            print("Sumbitting charge equilibration simulations...")
            await asyncio.gather(run_simulation_async(charge_dir, executor="sbatch"))

    print("Checking if all charge equilibration simulations are finished...")
    time_count = 0
    finished_dirs = []
    while True:
        flag = []
        unfinished_dirs = []
        for sub_dir in sorted(workdir.glob("*")):
            if not sub_dir.is_dir():
                continue
            elif sub_dir.name in finished_dirs:
                flag.append(True)
                continue
            charge_dir = sub_dir/"_charge_eq"
            if charge_dir.exists():
                if not (charge_dir/"Movies/System_0/").exists():
                    flag.append(False)
                    unfinished_dirs.append((sub_dir, "unfinished"))
                    continue
                else:
                    flag.append(True)
                    finished_dirs.append(sub_dir.name)
        print("{}/{}".format(sum(flag), len(flag)))
        time_count += interval
        if all(flag):
            print("charge equilibration simulations all done!")
            break
        elif time_count >= interval*10:
            print("charge equilibration simulations not finished after {} minites, force stop!".format(time_count/60))
            print("number of unfinished simulations: {}".format(len(unfinished_dirs)))
            unfinished_report = workdir/"unfinished_dirs_charge_eq_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            with open(unfinished_report, "w") as f:
                for d, reason in unfinished_dirs:
                    f.write("{}: {}\n".format(d, reason))
            print("information of unfinished simulations saved in {}".format(unfinished_report))
            break
        time.sleep(interval)
    unfinished_dirs_charge_eq = unfinished_dirs

    for sub_dir in workdir.glob("*"):
        if not sub_dir.is_dir():
            continue
        ## copy charged cif file to each task directory
        for task_name, press in task_tuples:
            task_dir = sub_dir/task_name
            charged_cifs = list((sub_dir/"_charge_eq/Movies/System_0/").glob("Framework_*_final_*.cif"))
            if len(charged_cifs) > 0:
                shutil.copy(charged_cifs[0], str(task_dir/(sub_dir.name + ".cif")))   ## cif file
                print("{} >> {}".format(charged_cifs[0], task_dir/(sub_dir.name + ".cif")))

        ## run helium void fraction simulations if necessary
        vf_dir = sub_dir/"_helium_void_fraction"
        if vf_dir.exists():
            print("Sumbitting helium void fraction simulations...")
            await asyncio.gather(run_simulation_async(vf_dir, executor="sbatch"))

    print("Checking if all helium void fraction simulations are finished...")
    time_count = 0
    finished_dirs = []
    while True:
        flag = []
        unfinished_dirs = []
        for sub_dir in sorted(workdir.glob("*")):
            if not sub_dir.is_dir():
                continue
            elif sub_dir.name in finished_dirs:
                flag.append(True)
                continue
            vf_dir = sub_dir/"_helium_void_fraction"
            if vf_dir.exists():
                data_files = list((vf_dir/"Output/System_0/").glob("*.data"))
                if len(data_files) == 0:
                    flag.append(False)
                    unfinished_dirs.append((sub_dir, "No data file found"))
                    continue
                with open(data_files[0], "r") as f:
                    content = f.read().strip()
                if "The end time was" in content:
                    flag.append(True)
                    finished_dirs.append(sub_dir.name)
                else:
                    flag.append(False)
                    unfinished_dirs.append((sub_dir, "unfinished"))

        print("{}/{}".format(sum(flag), len(flag)))
        time_count += interval
        if all(flag):
            print("helium void fraction simulations all done!")
            break
        elif time_count >= interval*10:
            print("helium void fraction simulations not finished after {} minites, force stop!".format(time_count/60))
            print("number of unfinished simulations: {}".format(len(unfinished_dirs)))
            unfinished_report = workdir/"unfinished_dirs_void_fraction_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            with open(unfinished_report, "w") as f:
                for d, reason in unfinished_dirs:
                    f.write("{}: {}\n".format(d, reason))
            print("information of unfinished simulations saved in {}".format(unfinished_report))
            break
        time.sleep(interval)
    unfinished_dirs_void_fraction = unfinished_dirs

    return unfinished_dirs_charge_eq, unfinished_dirs_void_fraction

def run_gcmc_tasks(workdir: Union[str, Path],
                    task_tuples: List[tuple],
                    executor="sbatch",
                    exclude_dirs: List[str] = [],
                    ):
    """
    executor: "bash", "nohup", or "sbatch", "sbatch_array"
    """
    workdir = Path(workdir)
    print("Submitting GCMC simulation tasks...")
    processes = []
    task_dirs = []
    with open(workdir/"gcmc_failed_tasks.txt", "w") as f:
        f.write("")
    for sub_dir in workdir.glob("*"):
        if (not sub_dir.is_dir()) or (sub_dir in exclude_dirs) or (sub_dir.name in exclude_dirs):
            continue
        vf = get_void_fraction(sub_dir/"_helium_void_fraction")
        # vf = 0.60
        for task_name, press in task_tuples:
            task_dir = sub_dir/task_name
            set_simulation_params(task_dir, vf=vf, temp=None, press=None, 
                            ff=None, unitcells=None, moldef=None,
                            init_cycles=None,
                            cycles=None,
                            framework_name=None,
                            )
            task_dirs.append(task_dir)
            if executor == "sbatch_array":
                continue
            processes.append(
                run_simulation(task_dir, executor=executor)
                )
            
    results = []
    time.sleep(10)
    if executor == "sbatch_array":
        process = run_simulation(workdir, executor="sbatch", script_name="run_mc_task")
        stdout, stderr = process.communicate()
        for task_dir in task_dirs:
            if not (task_dir/"Output/System_0/").exists() and (task_dir/"gcmc.err").exists():
                with open(workdir/"gcmc_failed_tasks.txt", "a") as f:
                    f.write("{}/{}\n".format(task_dir.parent.name, task_dir.name))
                with open(task_dir/"gcmc.err", "r") as f:
                    err = f.read()
                results.append((task_dir, err))
        return results

    for task_dir, proc in  zip(task_dirs, processes):
        data, err = proc.communicate()
        data = data.decode(encoding="utf-8", errors='ignore')
        err = err.decode(encoding="utf-8", errors='ignore')
        results.append((task_dir, err))
        if not (task_dir/"Output/System_0/").exists() and (task_dir/"gcmc.err").exists():
            with open(workdir/"gcmc_failed_tasks.txt", "a") as f:
                f.write("{}/{}\n".format(task_dir.parent.name, task_dir.name))
    return results


async def run_gcmc_tasks_async(workdir: Union[str, Path],
                        task_tuples: List[tuple],
                        executor="sbatch",
                        ):
    """
    executor: "bash", "nohup", or "sbatch"
    """
    workdir = Path(workdir)
    print("Submitting GCMC simulation tasks...")
    processes = []
    for sub_dir in workdir.glob("*"):
        if not sub_dir.is_dir():
            continue
        vf = get_void_fraction(sub_dir/"_helium_void_fraction")
        # vf = 0.60
        for task_name, press in task_tuples:
            task_dir = sub_dir/task_name
            set_simulation_params(task_dir, vf=vf, temp=None, press=None, 
                            ff=None, unitcells=None, moldef=None,
                            init_cycles=None,
                            cycles=None,
                            framework_name=None,
                            )
            processes.append(
                run_simulation_async(task_dir, executor=executor)
                )
    results = await asyncio.gather(*processes)
    return results


if __name__ == "__main__":

    pass

