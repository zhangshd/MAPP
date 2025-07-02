'''
Author: zhangshd
Date: 2024-09-09 21:24:41
LastEditors: zhangshd
LastEditTime: 2024-09-10 14:31:38
'''
from ase import io
from openbabel import pybel
import re
import pandas as pd
import os
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pyeqeq import run_on_cif

def cal_atom_charge(in_cif, overwite=True, out_cif=None, executable="pyeqeq", sanitize=True):
    """
    Calculate partial atom charges using EQeq method. 
    Reference: Wilmer C E, Kim K C, Snurr R Q. The Journal of Physical Chemistry Letters, 2012, 3(17): 2506–2511.
    executable: pyeqeq or openbabel
    """
    in_cif = str(in_cif)
    in_cif_clean = in_cif[:in_cif.rfind(".")] + "_clean.cif"
    if not out_cif:
        out_cif = in_cif[:in_cif.rfind(".")] + "_eqeq.cif"
    else:
        out_cif = str(out_cif)
    if os.path.exists(out_cif) and not overwite:
        print("Find existed file with calculated charge: {}".format(out_cif))
        return out_cif
    
    ## use ase to convert cif file in a standard way.
    # atoms = io.read(in_cif)
    # atoms.write(in_cif_clean, format="cif")
    if sanitize:
        in_cif_clean = standardized_cif(in_cif)
    else:
        in_cif_clean = in_cif
    print("write cleaned cif file: {}".format(in_cif_clean))
    

    if executable == "openbabel":
        mol = next(pybel.readfile("cif",in_cif_clean))
        ## call for EQeq methods implemented in openbabel
        charges = mol.calccharges("eqeq")

    # elif executable == "pyeqeq":
    #     out_str = run_on_cif(in_cif_clean, charge_precision=5, verbose=False, output_type="cif")
    #     with open(out_cif, "w") as f:
    #         f.write(out_str)
    #     print("The calculated charges have been writtern in {}".format(out_cif))
    #     return out_cif

    with open(in_cif_clean) as f:
        lines = f.read().strip().split("\n")
    flag = False
    atom_atrr_names = []
    atom_atrr_values = []
    fist_atom_row_num = 0
    for i,line in enumerate(lines):
        if line.strip().startswith("_atom_site"):
            atom_atrr_names.append(line.strip())
            if not flag:
                fist_atom_row_num = i
                flag = True
        elif flag and not line.strip().startswith("_"):
            atom_atrr_values.append(re.split("\s+", line.strip()))
        
    assert len(charges) == len(atom_atrr_values), "The number of charges({}) does not match the number of atoms({}).".format(len(charges), len(atom_atrr_values))

    new_atom_atrr_values = []
    if "_atom_site_charge" in atom_atrr_names:
        ch_index = atom_atrr_names.index("_atom_site_charge")
        for charge, atom_atrr_value in zip(charges, atom_atrr_values):
            atom_atrr_value[ch_index] = "{:.5f}".format(charge)
            new_atom_atrr_values.append(atom_atrr_value)
    else:
        atom_atrr_names.append("_atom_site_charge")
        for charge, atom_atrr_value in zip(charges, atom_atrr_values):
            atom_atrr_value.append("{:.5f}".format(charge))
            new_atom_atrr_values.append(atom_atrr_value)
                
    ## get head information from in_cif
    with open(in_cif) as f:
        lines = f.read().strip().split("\n")
    head_lines = []
    for line in lines:
        head_lines.append(line)
        if "_space_group_symop_operation_xyz" in line:
            break
        elif "_symmetry_equiv_pos_as_xyz" in line:
            head_lines[-1] = "  _space_group_symop_operation_xyz"
            break
    head_lines.append("  'x, y, z'\n\nloop_")
    ## write new cif file
    assert fist_atom_row_num > 0
    col_lengths = []
    df = pd.DataFrame(new_atom_atrr_values)
    for col in df.columns:
        col_lengths.append(df.loc[:,col].apply(lambda s: len(str(s))).max())
    with open(out_cif, "w") as f:
        f.write("\n".join(head_lines))
        f.write("\n  ")
        f.write("\n  ".join(atom_atrr_names))
        for values in new_atom_atrr_values:
            assert len(values) == len(col_lengths), "{} ≠ {}".format(len(values), len(col_lengths))
            f.write("\n  ")
            f.write("  ".join([values[i] + " "*(col_lengths[i] - len(values[i])) for i in range(len(values))]))
        f.write("\n")
    print("The calculated charges have been writtern in {}".format(out_cif))
    return out_cif

def standardized_cif(cif_file):
    cif_file = str(cif_file)
    print("*"*50)
    print(cif_file)
    parser = CifParser(cif_file)
    struct = parser.get_structures(primitive=False)[0]
    # Create SpacegroupAnalyzer object
    spacegroup_analyzer = SpacegroupAnalyzer(struct)
    # Get space group symbol
    space_group_symbol = spacegroup_analyzer.get_space_group_symbol()
    # print("Space Group Symbol:", space_group_symbol)
    # Get space group number
    space_group_number = spacegroup_analyzer.get_space_group_number()
    # print("Space Group Number:", space_group_number)
    # Get space group operations
    space_group_operations = spacegroup_analyzer.get_space_group_operations()
    # print("Space Group Operations:", space_group_operations)
    # Get standardized crystal structure (according to International Union of Crystallography standards)
    standardized_structure = spacegroup_analyzer.get_refined_structure()
    # print("Standardized Structure:", standardized_structure)
    out_file = cif_file.replace('.cif', '_clean.cif')
    standardized_structure.to(out_file, fmt='cif')
    print("standardized structure saved to: ", out_file)
    return out_file

