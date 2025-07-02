'''
Author: zhangshd
Date: 2024-09-09 21:24:42
LastEditors: zhangshd
LastEditTime: 2024-09-13 11:03:23
'''
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

from sklearn.cluster import KMeans
import shutil
import re
import subprocess
import stat
import asyncio
from sklearn import metrics
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial

week_abbrs = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
month_abbrs = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
time_patt = re.compile(r'(?:{})\s(?:{})'.format("|".join(week_abbrs), "|".join(month_abbrs)) + \
                       r"\s+\d{1,2}\s\d{2}:\d{2}:\d{2}\s\d{4}")
gases = ["CO2", "N2"]
units = ["mol/kg", "cm^3 (STP)/gr", "milligram/gram"]

loading_patterns = {}


molfrac_patterns = {g: re.compile(r".*\[{}\].*?MolFraction\:\s+(\d+\.*\d*)".format(g), 
                                  flags=re.MULTILINE|re.DOTALL) for g in gases}

qst_patterns = {
        "widom": re.compile("Average  <U_gh>_1-<U_h>_0.*\(\s+(\-*\d+\.*\d*)[\s\+\-/\d\.]+kJ/mol\)", flags=re.MULTILINE|re.DOTALL),
        "infinit dilution": re.compile("Total energy:\n[\=]+.*?Average\s+(\-*\d+\.*\d*)[\s\+\-/\d\.]+\[K\]", 
                                       flags=re.MULTILINE|re.DOTALL),
        "fluctuation formula": re.compile("Total enthalpy of adsorption\s+\-+.+Average[\s\d\.\+\-/]+\[K\][\s]+(\-*\d+\.*\d*)[\s\d\.\+\-/]+\[KJ/MOL\]", 
                                         flags=re.MULTILINE|re.DOTALL)
    }
for g in gases:
    qst_patterns[("fluctuation formula", g)] = re.compile("Enthalpy of adsorption component \d \[{}\]\s+\-+.+?Average[\s\d\.\+\-/]+\[K\][\s]+(\-*\d+\.*\d*)[\s\d\.\+\-/]+\[KJ/MOL\]".format(g), 
                                         flags=re.MULTILINE|re.DOTALL)
    for u in units:
        loading_patterns[(g, u)] = re.compile(
                r".*\[{}\].*?Average loading absolute \[{} framework\]\s+(\d+\.*\d*)".format(g, u) + \
                r".*?Average loading excess \[{} framework\]\s+(\-{}\d+\.*\d*)".format(u, "{,1}"), 
                flags=re.MULTILINE|re.DOTALL)

def extract_simulation_duration(content):
    mathed_ts = time_patt.findall(content)
    if len(mathed_ts)!= 2:
        return None
    start_time_str = mathed_ts[0]
    end_time_str = mathed_ts[1]

    # Convert the extracted time strings to datetime objects
    start_time = datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %Y")
    end_time = datetime.strptime(end_time_str, "%a %b %d %H:%M:%S %Y")

    # Calculate the duration
    duration = end_time - start_time

    # Format as number of hours
    formatted_duration = duration.total_seconds()/3600

    return formatted_duration


def calculate_rrmse(actual, predicted):
    """
    Calculate the Relative Root Mean Square Error (RRMSE).
    
    :param actual: array-like, the actual values
    :param predicted: array-like, the predicted values
    :return: float, the RRMSE
    """
    # Ensure that the inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Check that the dimensions match
    if actual.shape != predicted.shape:
        raise ValueError("The dimensions of actual and predicted do not match.")
    
    # Calculate the MSE
    mse = np.mean((actual - predicted) ** 2)
    
    # Calculate the mean of the actual values
    mean_actual = np.mean(actual)
    
    # Avoid division by zero
    if mean_actual == 0:
        raise ValueError("The mean of the actual values is zero, cannot compute RRMSE.")
    
    # Calculate the RRMSE
    rrmse = np.sqrt(mse) / mean_actual
    
    return rrmse

def sample_by_kmeans(df, columns4cluster, n_samples=20, random_state=42):
    print("df_shape: ", df.shape)
    df = df.copy()
    points = df[columns4cluster].values
    # Use K-Means algorithm to cluster these points into n_sample clusters
    # centroid, label = kmeans2(points, n_sample, seed=10)
    km = KMeans(n_clusters=n_samples, random_state=random_state)
    km.fit(points)
    # centroid = km.cluster_centers_
    df["SW_cluster"] = km.labels_
    sampled_dfs = []
    np.random.seed(random_state)
    for l in np.unique(km.labels_):
        # Find the indices of all points belonging to the current cluster
        sub_df = df[df["SW_cluster"]==l]
        rand_idx = np.random.permutation(len(sub_df))[0]
        sampled_dfs.append(sub_df.iloc[rand_idx:rand_idx+1])
    sampled_df = pd.concat(sampled_dfs)

    print("sampled_df_shape: ", sampled_df.shape)
    return sampled_df

def chxmod(file_path):
    file_path = str(file_path)
    # Retrieve current file permissions
    current_permissions = stat.S_IMODE(os.lstat(file_path).st_mode)
    # Add execution permissions: add execution permissions for user (u), group (g), and others (o)
    os.chmod(file_path, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def calculate_repetitions(cif_path, cutoff_radius):
    # Initialize unit cell parameters
    a = b = c = None
    alpha = beta = gama = None

    # Read the CIF file and extract unit cell parameters
    with open(cif_path, 'r') as file:
        for line in file:
            if line.startswith('_cell_length_a'):
                a = float(line.split()[1])
            elif line.startswith('_cell_length_b'):
                b = float(line.split()[1])
            elif line.startswith('_cell_length_c'):
                c = float(line.split()[1])
            elif line.startswith('_cell_angle_alpha'):
                alpha = float(line.split()[1])
            elif line.startswith('_cell_angle_beta'):
                beta = float(line.split()[1])
            elif line.startswith('_cell_angle_gamma'):
                gama = float(line.split()[1])

    

    # Ensure that all necessary parameters have been obtained
    if a is None or b is None or c is None:
        raise ValueError("Could not find all cell length parameters in the CIF file.")
    
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gama)

    # Calculate the volume of the unit cell
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 +
        2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Calculate the perpendicular length in each direction
    c_x = volume / (b * c * np.sin(alpha_rad))
    c_y = volume / (a * c * np.sin(beta_rad))
    c_z = volume / (a * b * np.sin(gamma_rad))

    # Calculate the number of unit cells needed in each direction
    nx = int(np.ceil((2 * cutoff_radius) / c_x))
    ny = int(np.ceil((2 * cutoff_radius) / c_y))
    nz = int(np.ceil((2 * cutoff_radius) / c_z))

    # Return the number of repetitions of the unit cell
    return nx, ny, nz


def set_raspa_dir(runfile, raspa_dir="/opt/share/RASPA/simulations"):
    with open(runfile, "r") as f:
        content = f.read()
    content = re.sub("(RASPA_DIR=).+", r"\1" + raspa_dir, content, flags=re.MULTILINE)
    with open(runfile, "w") as f:
        f.write(content)


def get_void_fraction(gcmc_dir, use_final=True):
    gcmc_dir = Path(gcmc_dir)
    outout_dir = gcmc_dir/"Output/System_0"
    data_files = list(outout_dir.glob("output_*.data"))
    if len(data_files) == 0:
        print("Found no valid result in {}".format(gcmc_dir))
        return None
    assert len(data_files) == 1, "There are mutilple output data in {}".format(outout_dir)
    data_file = data_files[0]
    with open(data_file) as f:
        content = f.read()
        void_fracs = re.findall(r"Rosenbluth factor new: (\d\.\d+)", content)
    if len(void_fracs) == 0:
        print("Found no valid result in {}".format(data_file))
        return None
    void_fracs = [float(v) for v in void_fracs]
    mean_void_frac = sum(void_fracs)/len(void_fracs)
    print("helium void fraction (mean): ", mean_void_frac)
    if not use_final:
        return mean_void_frac
    final_void_fracs = re.findall("\[helium\] Average Widom Rosenbluth-weight:\s+(\d+\.\d+)", content)
    assert len(final_void_fracs) == 1
    final_void_frac = float(final_void_fracs[0])
    print("helium void fraction (final): ", final_void_frac)
    return final_void_frac

def process_type(value, targer_type):
    if value is not None:
        value = targer_type(value)
    return value

def set_ff_params(def_file, cutoff_rule="truncated", tail_correction="no"):
    with open(def_file) as f:
        content = f.read()
    content = re.sub("(?<=# general rule for shifted vs truncated\n)(\w+)", cutoff_rule, content)
    content = re.sub("(?<=# general rule tailcorrections\n)(\w+)", tail_correction, content)
    with open(def_file, "w") as f:
        f.write(content)
def set_simulation_params(gcmc_dir, vf=None, temp=None, press=None, 
                          ff=None, unitcells=None, moldef=None,
                          init_cycles=None,
                          cycles=None,
                          framework_name=None,
                          components=None,
                          molfractions=None,
                          restartfile=None,
                          continue_after_crash=None,
                          write_crash_every=None, 
                          remove_atom_number_code_from_label=None,
                          verbose=0
                          ):
    gcmc_dir = Path(gcmc_dir)
    with open(gcmc_dir/"simulation.input") as f:
        content = f.read()

    patterns = [
        "(HeliumVoidFraction\s*)\s.+",
        "(ExternalTemperature\s*)\s.+",
        "(ExternalPressure\s*)\s.+",
        "(Forcefield\s*)\s.+", 
        "(UnitCells\s*)\s.+", 
        "(MoleculeDefinition\s*)\s.+", 
        "(NumberOfInitializationCycles\s*)\s.+", 
        "(NumberOfCycles\s*)\s.+", 
        "(FrameworkName\s*)\s.+",
        "(RestartFile\s*)\s.+",
        "(ContinueAfterCrash\s*)\s.+",
        "(WriteBinaryRestartFileEvery\s*)\s.+",
        "(RemoveAtomNumberCodeFromLabel\s*)\s.+",
    ]
    if verbose > 1:
        print("*"*50)
        print(gcmc_dir)
    if isinstance(press, (list, tuple)):
        press = [process_type(p, float) for p in press]
        press = sorted(press)
        press = " ".join([str(p) for p in press])
        ## if pressure is high, write binary restart file every 1000 cycles
        continue_after_crash = "yes"
        write_crash_every = 1000
    else:
        press = process_type(press, float)
        ## if pressure is high, write binary restart file every 1000 cycles
        if press is not None and press > 100000:
            continue_after_crash = "yes"
            write_crash_every = 1000
        else:
            content = re.sub("(WriteBinaryRestartFileEvery\s*)\s.+", "", content)

    params = [process_type(vf, float), process_type(temp, float), 
              press, ff, unitcells, moldef, 
              process_type(init_cycles, int), process_type(cycles, int), 
              framework_name, restartfile, continue_after_crash, process_type(write_crash_every, int), 
              remove_atom_number_code_from_label]
    for param, patern in zip(params, patterns):
        if param is None:
            continue
        content, n = re.subn(patern, r"\1 {}".format(param), content)
        if verbose > 1:
            print("Set {}: {}".format(patern.split(r"\s")[0]+")", param))
    
    with open(gcmc_dir/"simulation.input", "w") as f:
        f.write(content)
        f.write("\n\n")
    
    if components is None:
        return
    elif isinstance(components, str):
        components = [components]
        molfractions = [1]
    elif isinstance(components, (list, tuple)) and len(components) == 1:
        molfractions = [1]
    component_blks = re.findall(r'(Component\s+\d+.*?)(?=Component|\Z)', content, re.DOTALL)

    # print("{}\n{}\n{}".format(components, molfractions, component_blks))
    assert len(components) == len(molfractions) == len(component_blks), \
        "Number of components, molfractions, and component blocks not equal: {}\n{}\n{}".format(components, 
                                                                                                molfractions, 
                                                                                                component_blks)
    new_blocks = []
    for component, fraction, block in zip(components, molfractions, component_blks):
        if verbose > 1:
            print("Set component {} with fraction {}".format(component, fraction))
        new_block = re.sub("(MoleculeName\s*)\s.+", r"\1 {}".format(component), block)
        new_block = re.sub("(MolFraction\s*)\s.+", r"\1 {:.5f}".format(fraction), new_block)
        new_blocks.append(new_block)
    for block, new_block in zip(component_blks, new_blocks):
        content = content.replace(block, new_block)
    with open(gcmc_dir/"simulation.input", "w") as f:
        f.write(content)
        f.write("\n\n")
        

async def run_simulation_async(gcmc_dir, executor="nohup"):
    
    gcmc_dir = Path(gcmc_dir)
    process = await asyncio.create_subprocess_shell(
        f"{executor} {gcmc_dir/'run'} &",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
        cwd=str(gcmc_dir)
    )
    return process

def run_simulation(gcmc_dir, executor="sbatch", script_name="run"):
    gcmc_dir = Path(gcmc_dir)
    # 使用 subprocess.Popen 来同步执行子进程
    process = subprocess.Popen(
        f"{executor} {gcmc_dir/script_name}",
        # [executor, str(gcmc_dir/'run'), "&"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ.copy(),
        cwd=str(gcmc_dir)
    )
    return process

def get_adsorption_loading(gcmc_dir, components=["CO2"]):
    pattern = re.compile("".join(
        [r".*\[{}\].*?Average loading absolute \[mol/kg framework\]\s+(\d+\.*\d*)".format(c) + \
           r".*?Average loading excess \[mol/kg framework\]\s+(\-{,1}\d+\.*\d*)" for c in components]
    ), flags=re.MULTILINE|re.DOTALL)
    gcmc_dir = Path(gcmc_dir)
    outout_dir = gcmc_dir/"Output/System_0"
    data_files = list(outout_dir.glob("output_*.data"))
    if len(data_files) == 0:
        print("Found no valid result in {}".format(gcmc_dir))
        return [None, None]*len(components)
    assert len(data_files) == 1, "There are mutilple output data in {}".format(outout_dir)
    data_file = data_files[0]
    with open(data_file) as f:
        lines = []
        flag = False
        for line in f.readlines():
            if line.startswith("Number of molecules:"):
                flag = True
            elif line.startswith("Average Widom Rosenbluth factor"):
                flag = False
            if not flag:
                continue
            lines.append(line)
        content = "".join(lines)
    res = re.findall(pattern, content)
    if len(res) == 0:
        print("Found no valid result in {}".format(gcmc_dir))
        return [None, None]*len(components) 
    assert len(res) == 1, res
    print("#"*50)
    print("components: {}".format(components))
    print(res)
    print("#"*50)
    return [float(v) for v in res[0]]


def get_adsorption_isotherm(gcmc_dir, components=["CO2"], unit="mol/kg", verbose=2):
    """
    unit: mol/kg, milligram/gram, cm^3 (STP)/gr, cm^3 (STP)/cm^3
    """
    patterns = [re.compile(
        r".*\[{}\].*?Average loading absolute \[{} framework\]\s+(\d+\.*\d*)".format(c, unit) + \
        r".*?Average loading excess \[{} framework\]\s+(\-{}\d+\.*\d*)".format(unit, "{,1}"), 
        flags=re.MULTILINE|re.DOTALL) for c in components]
    molfrac_patterns = [re.compile(r".*\[{}\].*?MolFraction\:\s+(\d+\.*\d*)".format(c), 
                                  flags=re.MULTILINE|re.DOTALL) for c in components]
    gcmc_dir = Path(gcmc_dir)
    if verbose > 0:
        print("*"*50)
        print(components, gcmc_dir)
        print("*"*50)
    outout_dir = gcmc_dir/"Output/System_0"
    isotherm_res = []
    for data_file in outout_dir.glob("output_*.data"):
        press = float(data_file.name.split("_")[-1].replace(".data", ""))/100000
        temp = float(data_file.name.split("_")[-2])
        with open(data_file) as f:
            all_lines= f.readlines()
        lines = []
        time_lines = []
        flag = False
        invalid_flag = False
        for i, line in enumerate(all_lines):
            if line.strip().startswith("Simulation started on ") or line.strip().startswith("Simulation finished on"):
                ## add lines containing simulation start time or end time
                time_lines.append(all_lines[i-1].strip())
            elif "THE SIMULATION RESULTS ARE WRONG!!" in line or \
                "THERE ARE ATOM-PAIRS WITH NO VDW INTERACTION" in line:
                print("Found wrong simulation result in {}".format(data_file))
                invalid_flag = True
                break
            # elif "WARNING: THE SYSTEM HAS A NET CHARGE" in line:
            #     print("Found warning in simulation result in {}".format(data_file))
            #     invalid_flag = True
            elif line.strip().startswith("Number of molecules:"):
                flag = True
            elif line.strip().startswith("Average Widom Rosenbluth factor"):
                flag = False
            elif line.strip().startswith("MoleculeDefinitions:"):
                flag = True
            elif line.strip().startswith("Framework Status"):
                flag = False
            if not flag:
                continue
            lines.append(line)
        if invalid_flag:
            isotherm_res.append([None, temp, press, components, None, unit] + \
                                [None, None, None])
            continue
        content = "".join(lines)
        time_content = "".join(time_lines)
        simu_duration = extract_simulation_duration(time_content)
        for component, pattern, molfrac_pattern in zip(components, patterns, molfrac_patterns):
            res = re.findall(pattern, content)
            frac_res = re.findall(molfrac_pattern, content)
            if len(res) == 0:
                if verbose > 0:
                    print("Found no valid result in {}".format(data_file))
                isotherm_res.append([component, temp, press, components, None, unit] + \
                                    [None, None] + [simu_duration])
                continue
            assert len(res) == len(frac_res) == 1, (res, frac_res)
            if verbose > 1:
                print("#"*50)
                print("components: {}, unit: {}".format(component, unit))
                print(res)
                print("#"*50)
            isotherm_res.append([component, temp, press, "_".join(components), float(frac_res[0]), unit] + \
                                [float(v) for v in res[0]] + [simu_duration])
    return isotherm_res

def get_heat_isotherm(gcmc_dir, mode="widom", components=["CO2"], verbose=2):
    """
    mode: widom, infinit dilution, fluctuation formula
    """
    gcmc_dir = Path(gcmc_dir)
    if verbose > 0:
        print("*"*50)
        print(gcmc_dir)
        print("*"*50)
    outout_dir = gcmc_dir/"Output/System_0"
    heat_res = []
    for data_file in outout_dir.glob("output_*.data"):
        press = float(data_file.name.split("_")[-1].replace(".data", ""))/100000
        temp = float(data_file.name.split("_")[-2])
        with open(data_file) as f:
            all_lines= f.readlines()
        lines = []
        time_lines = []
        flag = False
        invalid_flag = False
        for i, line in enumerate(all_lines):
            if line.strip().startswith("Simulation started on ") or line.strip().startswith("Simulation finished on"):
                ## add lines containing simulation start time or end time
                time_lines.append(all_lines[i-1].strip())
            elif "THE SIMULATION RESULTS ARE WRONG!!" in line or \
                "THERE ARE ATOM-PAIRS WITH NO VDW INTERACTION" in line:
                print("Found wrong simulation result in {}".format(data_file))
                invalid_flag = True
                break
            elif line.strip().startswith("Enthalpy of adsorption:"):
                flag = True
            elif line.strip().startswith("derivative of the chemical potential with respect to density (constant T,V):"):
                flag = False
            elif line.strip() == "Total energy:":
                flag = True
            elif line.strip() == "Number of molecules:":
                flag = False
            elif line.strip().startswith("Average adsorption energy <U_gh>_1-<U_h>_0"):
                flag = True
            elif line.strip().startswith("Simulation finished,"):
                flag = False
            elif "(Adsorbate molecule)" in line:
                flag = True
            elif "Density of the bulk fluid phase:" in line:
                flag = False
            if not flag:
                continue
            lines.append(line)
        if invalid_flag:
            heat_res.append([None]*7)
            continue
        content = "".join(lines)
        time_content = "".join(time_lines)
        simu_duration = extract_simulation_duration(time_content)
        if len(components) >= 1 and mode == "fluctuation formula":
            for component in components:
                if len(components) > 1:
                    pattern = qst_patterns[("fluctuation formula", component)]
                else:
                    pattern = qst_patterns["fluctuation formula"]
                molfrac_pattern = molfrac_patterns[component]
                res = pattern.findall(content)
                frac_res = molfrac_pattern.findall(content)
                if len(res) == 0:
                    if verbose > 1:
                        print("Found no valid result for {} in {}".format(component, data_file))
                    heat_res.append([None]*7)
                    continue
                assert len(res) == len(frac_res) == 1, (res, frac_res)
                Qst = -float(res[0])
                if verbose > 1:
                    print("#"*50)
                    print("Gas: {}; T: {}; P: {}; Qst: {}; Mode: {}".format(component, temp, press, Qst, mode))
                    print("#"*50)
                heat_res.append([component, temp, press, "_".join(components), float(frac_res[0]), Qst, simu_duration])
            continue

        res = qst_patterns[mode].findall(content)
        if len(res) == 0:
            if verbose > 1:
                print("Found no valid result in {}".format(gcmc_dir))
            heat_res.append([None]*7)
            continue
        assert len(res) == 1, res
        ## Qst=-ΔH
        if mode == "infinit dilution":
            Qst = -((float(res[0]) - temp)*8.314462618/1000)  
        else:
            Qst = -float(res[0])
        if verbose > 1:
            print("#"*50)
            print("Gas: {}; T: {}; P: {}; Qst: {}; Mode: {}".format(components[0], temp, press, Qst, mode))
            print("#"*50)
        heat_res.append([components[0], temp, press, "_".join(components), 1.0, Qst] + [simu_duration])
    return heat_res

def get_heat_of_adsorption(gcmc_dir, mode="widom"):
    """
    mode: widom, inﬁnite dilution, ﬂuctuation formula
    """
    patterns = {
        "widom": re.compile("Average  <U_gh>_1-<U_h>_0.*\(\s+(\-*\d+\.*\d*)[\s\+\-/\d\.]+kJ/mol\)", flags=re.MULTILINE|re.DOTALL),
        "inﬁnite dilution": re.compile("Total energy:\n[\=]+.*?Average\s+(\-*\d+\.*\d*)[\s\+\-/\d\.]+\[K\]", 
                                       flags=re.MULTILINE|re.DOTALL),
        "ﬂuctuation formula": re.compile("Total enthalpy of adsorption\s+\-+.+Average[\s\d\.\+\-/]+\[K\][\s]+(\-*\d+\.*\d*)[\s\d\.\+\-/]+\[KJ/MOL\]", 
                                         flags=re.MULTILINE|re.DOTALL)
    }

    gcmc_dir = Path(gcmc_dir)
    outout_dir = gcmc_dir/"Output/System_0"
    data_files = list(outout_dir.glob("output_*.data"))
    if len(data_files) == 0:
        print("Found no valid result in {}".format(gcmc_dir))
        return None, None
    assert len(data_files) == 1, "There are mutilple output data in {}".format(outout_dir)
    data_file = data_files[0]
    pressure = float(data_file.name.split("_")[-1].split(".")[0])
    with open(data_file) as f:
        lines = []
        flag = False
        for line in f.readlines():
            if line.startswith("Enthalpy of adsorption:"):
                flag = True
            elif line.startswith("Simulation finished"):
                flag = False
            elif line.startswith("External temperature:"):
                lines.append(line)
            if not flag:
                continue
            lines.append(line)
        content = "".join(lines)
    res = re.findall(patterns[mode], content)
    if len(res) == 0:
        print("Found no valid result in {}".format(gcmc_dir))
        return None, None
    assert len(res) == 1, res
    if mode == "inﬁnite dilution":
        T = float(re.findall("External temperature:\s+(\d+\.*\d*)\s+\[K\]", content)[0])
        detaH = (float(res[0]) - T)*8.314462618/1000
    else:
        detaH = float(res[0])
    print("#"*50)
    print(detaH)
    print("#"*50)
    return detaH, pressure


def analyze_results(df, ref_col="CO2_uptake_1bar_298K (mmol/g)", 
                    target_cols=["pure_CO2_absolute", "pure_CO2_excess"],
                    show_metrics=True,
                    figsize=(12,8),
                    ):

    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # for target_col in target_cols:
    #     print("="*50)
    #     print(ref_col)
    #     print(target_col)
    #     values = df[[ref_col, target_col]].dropna().values
    #     r2 = metrics.r2_score(values[:, 0], values[:, 1])
    #     mape = metrics.mean_absolute_percentage_error(values[:, 0], values[:, 1])
    #     rrmse = calculate_rrmse(values[:, 0], values[:, 1])
    #     print("R2: {:.4f}; \nMAPE: {:.4f}\nRRMSE: {:.4f}".format(r2, mape, rrmse))
    #     plt.scatter(values[:, 0], values[:, 1], label=target_col, alpha=0.5)
    # plt.xlabel("ref results")
    # plt.ylabel("our results")
    # plt.legend()
    # plt.title(ref_col)
    # plt.show()
    # plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    for target_col in target_cols:
        print("="*50)
        print(ref_col)
        print(target_col)
        values = df[[ref_col, target_col]].dropna().values
        r2 = metrics.r2_score(values[:, 0], values[:, 1])
        rmse = np.sqrt(metrics.mean_squared_error(values[:, 0], values[:, 1]))
        rrmse = calculate_rrmse(values[:, 0], values[:, 1])
        print("R2: {:.4f}; \nRMSE: {:.4f}\nRRMSE: {:.4f}".format(r2, rmse, rrmse))

        if len(target_cols) == 1:
            label = None
        else:
            label = target_col
        ax.scatter(values[:, 0], values[:, 1], label=label, alpha=0.5)

        max_value = values.max()
        min_value = values.min()
        offset = (max_value*0.05)
        
        ax.set_xlim(min_value - offset, max_value + offset)
        ax.set_ylim(min_value - offset, max_value + offset)

        ax.plot([min_value, max_value], [min_value, max_value], 'r--') 

        if show_metrics:
            ax.text(max_value - offset*6, min_value + offset, 
                    "R2: {:.4f}\nRMSE: {:.4f}\nRRMSE: {:.4f}".format(r2, rmse, rrmse), 
                    fontsize=12, color='red')
    plt.xlabel("ref results")
    plt.ylabel("our results")
    plt.legend()
    plt.title(ref_col)
    return fig, ax


def get_json_heat_data(file):
    """
    extract heat of adsoption data in NIST-format-like json file from MOFXDB.
    """
    with open(file) as f:
        dic = json.load(f)
    records = []
    heats = dic["heats"]
    for heat in heats:
        mof = heat["adsorbent"]["name"]
        doi = heat["DOI"]
        simin = heat["simin"]
        T = heat["temperature"]
        pressureUnits = heat["pressureUnits"]
        adsorptionUnits = heat["adsorptionUnits"]
        compositionType = heat["compositionType"]
        molecule_forcefield = heat["molecule_forcefield"]
        adsorbent_forcefield = heat["adsorbent_forcefield"]
        category = heat["category"]
        for isotherm_data in heat["isotherm_data"]:
            P = isotherm_data["pressure"]
            for specie_data in isotherm_data["species_data"]:
                gas_name = specie_data["name"]
                adsorption = specie_data["adsorption"]
                composition = specie_data["composition"]
                records.append([
                    mof, 
                    gas_name, 
                    adsorption, 
                    adsorptionUnits, 
                    T, 
                    P, 
                    pressureUnits,
                    composition, 
                    compositionType,
                    molecule_forcefield,
                    adsorbent_forcefield,
                    category,
                    simin,
                    doi
                    ])
    return records

def get_adsorption_df(task_dir, gases, unit, verbose):
    data_columns = ["GasName", 
                    "Temperature[K]", 
                    "Pressure[bar]", 
                    "AllComponents",
                    "MoleculeFraction",
                    "LoadingUnit", 
                    "AbsLoading", 
                    "ExcessLoading", 
                    "SimuDuration[h]"
                    ]
    task_dir = Path(task_dir)
    components = re.findall("|".join(gases), task_dir.name)
    isotherm_res = get_adsorption_isotherm(task_dir, components=components, unit=unit, verbose=verbose)
    df = pd.DataFrame(isotherm_res, columns=data_columns)
    df.insert(0, "MofName", task_dir.parent.name)
    df["Notes"] = task_dir.parent.parent.name
    df.sort_values(by=["Temperature[K]", "Pressure[bar]"], inplace=True)
    df.dropna(subset="AbsLoading", inplace=True)
    return df

def process_isotherm_results(test_dir, gases,
                            unit="milligram/gram",
                            verbose=0,
                            n_jobs=8
                            ):
    """
    Get specific isotherm data from GCMC simulations output files.

    Args:
    test_dir (str): Directory containing the GCMC simulation output files.
    gases (list): List of gases for which isotherm data needs to be extracted.
    task_names (list): List of task names corresponding to the GCMC simulations.
    data_columns (list): List of column names for the isotherm data.

    Returns:
    pandas.DataFrame: Concatenated DataFrame containing the extracted isotherm data for the specified gases.
    """
    
    dfs = []
    test_dir = Path(test_dir)
    task_dirs = []
    for sub_dir in test_dir.glob("*"):
        if not sub_dir.is_dir():
            continue
        for task_dir in sub_dir.glob("*"):
            if not task_dir.is_dir():
                continue
            elif "Adsorption" not in task_dir.name:
                continue
            outout_dir = task_dir/"Output/System_0"
            data_files = list(outout_dir.glob("output_*.data"))
            if len(data_files) == 0:
                print("No valid results found in: ")
                print(task_dir)
                continue
            task_dirs.append(task_dir)
    if len(task_dirs) == 0:
        return pd.DataFrame([])
    get_adsorption_df_p = partial(get_adsorption_df, gases=gases, unit=unit, verbose=verbose)
    with mp.Pool(processes=min(n_jobs, len(task_dirs))) as pool:
        dfs = pool.map(get_adsorption_df_p, task_dirs)
    if len(dfs) == 0:
        return pd.DataFrame([])
    return pd.concat(dfs)

def get_heat_df(task_dir, gases, verbose):
    data_columns = [
                "GasName",
                "Temperature[K]", 
                "Pressure[bar]",
                "AllComponents",
                "MoleculeFraction",
                "Qst",
                "SimuDuration[h]"
                ]
    task_dir = Path(task_dir)
    components = re.findall("|".join(gases), task_dir.name)
    ## get mode of heat extraction
    if "Qst_widom" in task_dir.name:
        mode = "widom"
    elif "Qst_ideal" in task_dir.name:
        mode = "infinit dilution"
    else:
        mode = "fluctuation formula"
    isotherm_res = get_heat_isotherm(task_dir, mode=mode, components=components, verbose=verbose)
    df = pd.DataFrame(isotherm_res, columns=data_columns)
    df.insert(0, "MofName", task_dir.parent.name)
    df.sort_values(by=["Temperature[K]", "Pressure[bar]"], inplace=True)
    df["QstMethod"] = mode
    df["Notes"] = task_dir.parent.parent.name
    df.dropna(subset="Qst", inplace=True)

    return df

def process_heat_results(test_dir, 
                         gases,
                         verbose=0,
                         n_jobs=8
                         ):
    """
    Get specific heat of adsorption data from GCMC simulations output files.
    Args:
    test_dir (str): Directory containing simulation output files.
    gases (list): List of gases for which specific heat of adsorption data is to be extracted.
    task_names (list): List of task names associated with the simulations.
    mode (str, optional): Mode of heat extraction, default is "widom". other options are "inﬁnite dilution" and "ﬂuctuation formula".
    data_columns (list, optional): List of column names for the specific heat data, 
                                   default includes "Temperature[K]", "Pressure[bar]", "Qst", "SimuDuration[h]".
    Returns:
    pandas.DataFrame: Concatenated dataframes containing specific heat of adsorption data.
    """
    
    test_dir = Path(test_dir)
    dfs = []
    task_dirs = []
    for sub_dir in test_dir.glob("*"):
        if not sub_dir.is_dir():
            continue
        for task_dir in sub_dir.glob("*"):
            # print(task_dir)
            if not task_dir.is_dir():
                continue
            elif not ("Qst" in task_dir.name or "Adsorption" in task_dir.name):
                continue
            outout_dir = task_dir/"Output/System_0"
            data_files = list(outout_dir.glob("output_*.data"))
            if len(data_files) == 0:
                print("No valid results found in: ")
                print(task_dir)
                continue
            task_dirs.append(task_dir)
    if len(task_dirs) == 0:
        return pd.DataFrame([])
    get_heat_df_p = partial(get_heat_df, gases=gases, verbose=verbose)
    with mp.Pool(processes=min(n_jobs, len(task_dirs))) as pool:
        dfs = pool.map(get_heat_df_p, task_dirs)
    if len(dfs) == 0:
        return pd.DataFrame([])
    return pd.concat(dfs)

def convert_unit(adsorbate, value, unit, out_unit="cm3(STP)/g"):
    """
    Convert the unit of adsorption data to the desired unit (mmol/g).
    """
    mass_map = {
        "Carbon Dioxide": 44.01,
        "CO2": 44.01,
        "Nitrogen": 28.01,
        "N2": 28.01,
        "Oxygen": 31.99,
        "O2": 31.99,
        "Hydrogen": 2.016,
        "H": 2.016,

    }
    convert_coefi_map = {
        "mmol/g": 1,
        "cm3(STP)/g": 22.414,
        "mol/kg": 1,
        "mg/g": mass_map[adsorbate]/1000,
    }

    try:
        value = float(value)
    except Exception as e:
        return None
    if unit == "mmol/g":
        new_value = value
    elif unit == 'cm3(STP)/g':
        new_value = value / 22.414
    elif unit == 'ml/g':
        new_value = value / 22.414
    elif unit == 'mmol/kg':
        new_value = value / 1000
    elif unit == 'mol/kg':
        new_value = value
    elif unit == 'mg/g':
        new_value = value / mass_map[adsorbate]
    elif unit == 'mg/kg':
        new_value = value / (mass_map[adsorbate]*1000)
    elif unit == 'mol/g':
        new_value = value * 1000
    elif unit == 'g/g':
        new_value = value * 1000 / mass_map[adsorbate]
    else:
        return None
    return new_value * convert_coefi_map[out_unit]

