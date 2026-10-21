#!/usr/bin/env python
# coding: utf-8

# # GetProperties
# Jupyter notebook interface for the automated collection of molecular descriptors and post-processing (i.e., Boltzmann average, min/max values, etc.) from Gaussian16 log files. Be sure to use the included environment.

# ## Imports

# In[ ]:


# stdlib
import os
import time
import logging
import itertools

from pathlib import Path
from pprint import pprint
from multiprocessing import Pool

# Data wrangling
import numpy as np
import pandas as pd

# Misc
from rdkit import Chem
from natsort import natsorted
from IPython.display import SVG, display

# Custom
from GetProperties.goodvibes_functions import get_goodvibes_data
from GetProperties.property_functions import get_frontierorbs, get_polarizability, get_dipole, get_volume
from GetProperties.property_functions import get_SASA, get_nbo, get_nmr, get_distance, get_angles, get_dihedral
from GetProperties.property_functions import get_vbur_scan, get_sterimol_morfeus, get_chelpg, get_hirshfeld
from GetProperties.property_functions import get_pyramidalization, get_wall_time, get_plane_angle, get_sterimol_dbstep

from GetProperties.utils import convert_files_in_directory
from GetProperties.utils import _read_in_mol_sdf_with_xyz_correction
from GetProperties.utils import FILE_COLUMN_NAME
from GetProperties.utils import draw_3D_mol
from GetProperties.utils import configure_logger
from GetProperties.utils import split_compound_name
from GetProperties.utils import _get_atom_map, mol_to_image

# Initial settings
pd.options.display.max_columns = None
logger = logging.getLogger('notebook')


# ## Settings
# In this section, point the `data_dir` variable to the directory containing your G16 log files, and point `results_dir` 
# to the directory where results will be saved. The script will automatically convert the log files to `.mol` files for 
# substructure matching and atom mapping.

# In[ ]:


# Point to path of the log files
data_dir = Path('./data/sept15_allcurrentlogfiles/FINALALLLOGFILES/')

# Point to path where results and intermediate files are saved
results_dir = Path('./results/sept17pmFINAL')
results_dir.mkdir(exist_ok=True, parents=True)

# Define number of processors to use (set to integer or os.cpu_count())
procs = os.cpu_count()

# Turn on debug printing
debug = False

# Format the logging (you don't have to edit this)
configure_logger(debug=debug)


# ## Convert .log files to .mol files
# This section converts the `.log` files to `.mol` files, which can be read with RDKit for downstream processing. You do 
# not have to edit this cell.

# In[ ]:


# Convert the logfiles into .mol files for reading in with RDKit
# This function returns the files that failed
failed_files = convert_files_in_directory(directory=data_dir)

# Move and log the failed files
if len(failed_files) != 0:
    logger.warning(f'Some files do not have corresponding .mol files')
    for f in failed_files:
        logger.warning(f.name)

mol_files = [x for x in data_dir.glob('*.mol')]

logger.info('There are %d files that do not have a corresponding .mol file.', len(failed_files))
logger.info('There are %d .mol files.', len(mol_files))


# ## Define substructure for featurization
# Provide and [SMARTS](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) string or SMILES string that defines the substructure to featurize. If you are unfamiliar with SMARTS patterns, begin by copying the SMILES directly from ChemDraw first. If that does not correspond to your desired substructure (e.g., hydrogens are missing), consult the [SMARTS documentation](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html). You can visualize your
# SMARTS query with the [SMARTS PLUS](https://smarts.plus/) webtool.
# 

# In[ ]:


# Define a substructure common to all data files you are looking at
substructure = Chem.MolFromSmarts('CC=C[#6,#16]')

im = mol_to_image(substructure,
                  show_atom_indices=True,
                  image_size=(350, 350))

display(SVG(data=im))


# ## Skip files with multiple olefins, or multiple substructure matches
# Some structures may contain more than one instance of the substructure (e.g., multiple olefins). Since `_get_atom_map` 
# expects a single, unambiguous match, files with more than one match are moved out of `data_dir` into a subfolder so that 
# every downstream cell (which re-globs `data_dir`) automatically skips them. You do not have to edit this cell.

# In[ ]:


# Exclusion folder, where flagged files will be moved so they're excluded from all downstream data_dir.glob('*.mol') calls
multi_match_dir = data_dir / 'excluded_multiple_olefins'
multi_match_dir.mkdir(exist_ok=True)

skipped_multi_match = []

for mol_file in natsorted(list(data_dir.glob('*.mol'))):
    mol = Chem.MolFromMolFile(str(mol_file), removeHs=False)

    if mol is None:
        # Let the existing failed-match handling in the next cell deal with unreadable files
        continue

    # Note that substructure is listed in cell above
    matches = mol.GetSubstructMatches(substructure, uniquify=True)
    if len(matches) > 1:
        logger.warning('%s has %d matches to the substructure (e.g., multiple olefins) - moving to %s/',
                        mol_file.name, len(matches), multi_match_dir.name)
        skipped_multi_match.append(mol_file.name)
        mol_file.rename(multi_match_dir / mol_file.name)

logger.info('Skipped %d files with multiple substructure matches. They have been moved to %s.',
            len(skipped_multi_match), multi_match_dir)

# Save the list of excluded files to Excel
excluded_df = pd.DataFrame({FILE_COLUMN_NAME: skipped_multi_match})
excluded_df.to_excel(results_dir / 'excluded_multiple_olefins.xlsx', engine='openpyxl', index=False)


# ## Generate DataFrame containing files and atom numbers
# Here we read in the `.mol` files and identify the substructure in each. Identified atoms patterns will be 
# assigned a generic column label that we will redefine with a meaningful text label. You do not have to edit this cell.

# In[ ]:


# Get the atom maps in parallel. This returns a list of maps where the first item
# in the returned list is the file name and the remaining items are 1-indexed atom numbers
with Pool(processes=procs) as p:
    results = p.starmap(_get_atom_map, zip(natsorted(list(data_dir.glob('*.mol'))), itertools.repeat(substructure)))

failed_files = []
successful_files = []

# Iterate through the results (list of lists)
for item in results:

    # If the "item" is a failed match object, it should be of length 1 (just the file name)
    if len(item) == 1:
        logger.warning('%s failed substructure matching or was unreadable by RDKit.', item[0])
        failed_files.append(item[0])
    else:
        successful_files.append(item)

# Place the atom numbers for the substructure for each log file into a dataframe
prelim_df = pd.DataFrame(successful_files)
prelim_df.rename(columns={0: FILE_COLUMN_NAME, **{i: f'atom_{i}' for i in prelim_df.columns if i != 0}}, inplace=True)

# Save the prelim dataframe
prelim_df.to_csv(results_dir / 'preliminary_df.csv', encoding='utf-8-sig')

# Print the failed items
if len(failed_files) != 0:
    logger.warning('Files that were not read in with RDKit:')
    for f in failed_files:
        logger.warning(f)

logger.info('There are %d files that were not read in.', len(failed_files))
logger.info('There are %d files that successfully read in.', prelim_df.shape[0])
display(prelim_df)


# ## Rename columns to user-interpretable labels
# 
# This cell will sample rows in your `prelim_df` and display the molecules with atom labels (1-idnexed). These column headers in `prelim_df` should be reassigned to a user-defined label that is more meaningful for property collection. The dictionary template printed at the end of this cell should look something like `{'atom_1': '', 'atom_2': '', 'atom_3': '', ...}`. This can be used in the next cell by remapping the numbers to text labels that are easier to understand.
# 
# ```python
# atom_labels = {
#     'atom_1': 'C3',
#     'atom_2': 'C2prime',
#     'atom_3': 'C1prime',
#     'atom_4': 'N',
#     'atom_5': 'C1',
#     'atom_6': 'C2'
# }
# ```

# In[ ]:


# Define the number of structures you want to visualize
number_to_sample = 1

# Iterate over the df
for i, row in prelim_df.iterrows():

    if i == number_to_sample:
        break

    # Read in the file as mol object
    file = Path(data_dir / row[FILE_COLUMN_NAME]).with_suffix('.mol')
    file, mol = _read_in_mol_sdf_with_xyz_correction(file=file)

    logger.info(file.name)
    draw_3D_mol(mol=mol)

    if mol is None:
        logger.error('Sampled a None mol for %s', file.name)
        continue

# Print out a template dictionary
logger.info('Use this dictionary template to define your named atom map.')
pprint({k: '' for k in prelim_df.columns if k != FILE_COLUMN_NAME})


# ## Resolve asymmetric C1 vs C1prime by Hirsh_charge 

# In[ ]:


# Resolve which end of the SMARTS match (atom_1 vs atom_4) should be called C1prime, using the ENSEMBLE Boltzmann-weighted Hirshfeld charge -- not a single conformer's value.

# Work on a copy so prelim_df isn't touched until we're sure of the swap
resolve_df = prelim_df.copy(deep=True)

# Step 1: identify each compound before filtering
separator = '_'
resolve_df['PREFIX'] = resolve_df[FILE_COLUMN_NAME].apply(split_compound_name, delimiter=separator,return_key=0,)

# Step 2: per-conformer Hirshfeld charge on each candidate atom
resolve_df = get_hirshfeld(resolve_df, atom_list=['atom_1', 'atom_4'], data_dir=data_dir,procs=procs,)

# Step 3: per-conformer free energy for Boltzmann weighting
resolve_df = get_goodvibes_data(resolve_df,data_dir=data_dir,temp=298.15,procs=procs,)
energy_col_header = 'G(T)_spc(Hartree)'
temperature = 298.15
R = 1.9872036e-3
HARTREE_TO_KCAL = 627.5094740631

def boltzmann_weighted_avg(group,value_col,energy_col=energy_col_header,temp=temperature,):
    g = group.copy()
    g[value_col] = pd.to_numeric(g[value_col], errors='coerce')
    g[energy_col] = pd.to_numeric(g[energy_col], errors='coerce')
    g = g.dropna(subset=[value_col, energy_col])
    if g.empty:
        return np.nan
    rel_energy_kcal = (g[energy_col] - g[energy_col].min()) * HARTREE_TO_KCAL
    weights = np.exp(
        -rel_energy_kcal.to_numpy(dtype=float) / (R * temp))
    weights /= weights.sum()
    return float(
        np.sum(weights * g[value_col].to_numpy(dtype=float)))

# Step 4: ensemble Boltzmann-weighted charges per compound
boltz_charges = (resolve_df.groupby('PREFIX').apply(
        lambda g: pd.Series({
            'Hirsh_charge_atom_1_Boltz': boltzmann_weighted_avg(g,'Hirsh_charge_atom_1',),
            'Hirsh_charge_atom_4_Boltz': boltzmann_weighted_avg(g,'Hirsh_charge_atom_4',),})))

# Step 5: decide which compounds need to be swapped, atom_4 is intended to be the more positive C1prime side.
boltz_charges['swap_needed'] = (boltz_charges['Hirsh_charge_atom_1_Boltz'] > boltz_charges['Hirsh_charge_atom_4_Boltz'])
missing_charge_mask = boltz_charges[['Hirsh_charge_atom_1_Boltz', 'Hirsh_charge_atom_4_Boltz',]].isna().any(axis=1)

# Explicitly prevent missing values from causing a swap
boltz_charges.loc[missing_charge_mask, 'swap_needed'] = False
n_missing = int(missing_charge_mask.sum())

if n_missing:
    logger.warning('%d compound(s) had missing Hirshfeld/energy data and were NOT swapped (defaulted to atom_4 = C1prime).',n_missing,)

n_compounds_to_swap = int(boltz_charges['swap_needed'].sum())

logger.info(
    '%d of %d non-symmetric compounds need atom_1/atom_4 '
    '(and atom_2/atom_3) swapped so that C1prime is always the '
    'ensemble-averaged, more positive Hirshfeld-charge carbon.',
    n_compounds_to_swap,
    len(boltz_charges),)

# Charges before the swap
display(boltz_charges)

# Step 6: apply the compound-level decision to every conformer.
prelim_df['PREFIX'] = prelim_df[FILE_COLUMN_NAME].apply(split_compound_name, delimiter=separator, return_key=0,)

swap_mask = (prelim_df['PREFIX'].map(boltz_charges['swap_needed']).fillna(False).astype(bool))

prelim_df.loc[swap_mask,['atom_1', 'atom_4'],] = prelim_df.loc[swap_mask,['atom_4', 'atom_1'],].to_numpy()

prelim_df.loc[swap_mask,['atom_2', 'atom_3'],] = prelim_df.loc[swap_mask,['atom_3', 'atom_2'],].to_numpy()

prelim_df.drop(columns=['PREFIX'], inplace=True,)

logger.info(
    'Swapped atom_1<->atom_4 and atom_2<->atom_3 for %d conformers '
    'belonging to %d compounds.',
    int(swap_mask.sum()),
    n_compounds_to_swap,)

display(prelim_df)

# Step 7: display the Boltzmann-weighted charges after the swap. This is a display copy; the original boltz_charges table is unchanged.
boltz_charges_after_swap = boltz_charges.copy()

swap_rows = (boltz_charges_after_swap['swap_needed'].fillna(False).astype(bool))

boltz_charges_after_swap.loc[swap_rows,['Hirsh_charge_atom_1_Boltz', 'Hirsh_charge_atom_4_Boltz',],] = boltz_charges_after_swap.loc[swap_rows,['Hirsh_charge_atom_4_Boltz','Hirsh_charge_atom_1_Boltz',],].to_numpy()
display(boltz_charges_after_swap)

# Step 8: save the updated (post-swap) atom mapping to Excel
atom_mapping_cols = [FILE_COLUMN_NAME, 'atom_1', 'atom_2', 'atom_3', 'atom_4']
atom_map_export = prelim_df[atom_mapping_cols].copy()

# Flag which conformers were swapped, for traceability (optional but handy)
atom_map_export['swapped'] = (prelim_df[FILE_COLUMN_NAME].apply(split_compound_name, delimiter=separator, return_key=0).map(boltz_charges['swap_needed']).fillna(False).astype(bool))

atom_map_export.to_excel(results_dir / 'atom_map_post_swap.xlsx',index=False,)

logger.info('Saved updated (post-swap) atom mapping to %s',results_dir / 'atom_map_post_swap.xlsx',)

display(atom_map_export)


# ## Set an atom label dictionary
# Use this cell to assign the mapped atom numbers in the previous cell to interpretable labels. Ensure that your atom 
# numbers are correct by comparing the atom number assigned in the DataFrame to the atom number in a visualization program like PyMOL, Avogadro, or the 3D viewer above. 
# 
# Atom numbering of different conformers of the same compound should be the same. Symmetric substructures should be 
# processed in later steps to average descriptors or compute differences.
# 
# **Assign these correctly. Otherwise, the properties you collect will be for the wrong atoms.** 

# In[ ]:


# Create an atom label dictionary
atom_labels = {'atom_1': 'C1',
               'atom_2': 'CC2',
               'atom_3': 'CC2prime',
               'atom_4': 'C1prime'}

# Rename columns using the user input above
atom_map_post_swap = prelim_df.rename(columns=atom_labels)

# Drop unmapped atoms
unlabeled_columns = [x for x in atom_map_post_swap.columns if 'atom_' in x]
if unlabeled_columns:
    logger.warning('Dropping unlabeled atoms in your mapping. %s', unlabeled_columns)
    atom_map_post_swap.drop(columns=unlabeled_columns, inplace=True)
display(atom_map_post_swap)

# Remove specific atoms if you wish to not collect descriptors for particular atoms
atom_map_post_swap.drop(columns=[], inplace=True)

# df is what properties will be appended to, this creates a copy so that you have the original preserved
df = atom_map_post_swap.copy(deep=True)

all_mapped_atoms = list(atom_labels.values())

# Print a list of all atom labels that can be pasted into the
# functions to collect properties for all mapped atoms
logger.info('List of all mapped atoms in your substructure.')
logger.info(list(atom_labels.values()))

# Save atom map to Excel
atom_map_post_swap.to_csv(results_dir / 'finalnamed_atom_map_post_swap.csv', encoding='utf-8-sig', index=False)


# ## Define Properties to Collect
# This cell contains all of the functions that extract/compute properties from the logfiles. If you do not wish to extract a property, place a pound sign `#` to comment out the function call.

# In[ ]:


t1 = time.time()
df = atom_map_post_swap.copy()

#---------------GoodVibes Energies----------------
# Uses the GoodVibes to compute the quasi harmonic corrected G(T)
# and single point corrected G(T) and other thermodynamic properties
df = get_goodvibes_data(df, data_dir=data_dir, temp=298.15, procs=procs)

#---------------Frontier Orbitals-----------------
# Extracts the E(HOMO), E(LUMO), mu(chemical potential or negative of molecular electronegativity),
# eta(hardness/softness), omega(electrophilicity index)
df = get_frontierorbs(df, data_dir=data_dir, procs=procs)

#---------------Polarizability--------------------
# Extracts the isotropic and anisotropic polarizability
df = get_polarizability(df, data_dir=data_dir, procs=procs)

#---------------Dipole----------------------------
# Extracts the total dipole moment magnitude in Debye
df = get_dipole(df, data_dir=data_dir, procs=procs)

#---------------Volume----------------------------
# Extracts the molar volume
# Requires the Gaussian keyword "volume" in the .com file
df = get_volume(df, data_dir=data_dir, procs=procs)

#---------------SASA------------------------------
# Uses MORFEUS to compute solvent accessible surface area (SASA) and the volume under the SASA
df = get_SASA(df, data_dir=data_dir, procs=procs)

#---------------NBO-------------------------------
# Extracts the natural charge from natural population analysis
# Requires the Gaussian keyword = "pop=nbo7" in the .com file
nbo_list = all_mapped_atoms
df = get_nbo(df, atom_list=nbo_list, data_dir=data_dir, procs=procs)

#---------------NMR-------------------------------
# Extracts the isotropic NMR shielding magnitude (inversely related to chemical shift)
# Requires the Gaussian keyword = "nmr=giao" in the .com file
nmr_list = all_mapped_atoms
df = get_nmr(df, atom_list=nmr_list, data_dir=data_dir, procs=procs)

#---------------ChelpG----------------------------
# Extracts the ChelpG ESP charge
# Requires the Gaussian keyword = 'pop=chelpg'
# a_list = all_mapped_atoms
# df = get_chelpg(df, atom_list=a_list, data_dir=data_dir, procs=procs)

#---------------Hirshfeld-------------------------
# Extracts the Hirshfeld charge, CM5 charge, Hirshfeld atom dipole
# Requires the Gaussian keyword = 'pop=hirshfeld'
a_list = all_mapped_atoms
df = get_hirshfeld(df, atom_list=a_list, data_dir=data_dir, procs=procs)

#---------------Dihedral--------------------------
# Computes the dihedral angle between 4 atoms
dihedral_list_of_lists = [['C1', 'CC2', 'CC2prime', 'C1prime']]
df = get_dihedral(df, dihedral_list=dihedral_list_of_lists, data_dir=data_dir, procs=procs)

#---------------Distance--------------------------
# Computes the distance between 2 atoms
dist_list_of_lists = [['C1', 'CC2'], ['CC2', 'CC2prime'], ['CC2prime', 'C1prime']]
df = get_distance(df, dist_list=dist_list_of_lists, data_dir=data_dir, procs=procs)

#---------------Angle-----------------------------
# Computes the angle between 3 atoms
angle_list_of_lists = [['C1', 'CC2', 'CC2prime'], ['CC2', 'CC2prime', 'C1prime']]
df = get_angles(df, angle_list=angle_list_of_lists, data_dir=data_dir, procs=procs)

#---------------Vbur Scan-------------------------
# Uses MORFEUS to compute the buried volume at a series of radii (including hydrogens)
# If you only want a single radius, put the same value for start_radius and end_radius (keep step_size > 0)
vbur_list = all_mapped_atoms
df = get_vbur_scan(df,
                   data_dir=data_dir,
                   a_list=vbur_list,
                   start_r=2.0,
                   end_r=4.5,
                   step_size=0.5,
                   procs=procs)

#---------------Pyramidalization------------------
# Uses MORFEUS to calculate pyramidalization of 3 atoms in closest proximity to the defined atom
# and collects two definitions of pyramidalization
# See https://kjelljorner.github.io/morfeus/pyramidalization.html
pyr_list = ['C1', 'CC2', 'CC2prime', 'C1prime']
df = get_pyramidalization(df, atom_list=pyr_list, data_dir=data_dir, procs=procs)

#---------------Sterimol MORFEUS------------------
# Uses MORFEUS to calculate Sterimol L, B1, and B5 values
# This is much faster than using DBSTEP and is recommeded unless you need Sterimol2Vec
sterimol_list_of_lists = [['C1', 'CC2'], ['CC2', 'CC2prime'], ['CC2prime', 'C1prime'], ['CC2', 'C1'], ['CC2prime', 'CC2'], ['C1prime', 'CC2prime']]
df = get_sterimol_morfeus(df, data_dir=data_dir, sterimol_list=sterimol_list_of_lists, radius=None, procs=procs)

#---------------Buried Sterimol-------------------
# Uses MORFEUS to calculate Sterimol L, B1, and B5 values within a sphere
# Atoms outside the sphere + (0.5*vdW_radius) are removed before computing Sterimol
# See https://kjelljorner.github.io/morfeus/sterimol.html
sterimol_list_of_lists = [['C1', 'CC2'], ['CC2', 'CC2prime'], ['CC2prime', 'C1prime'],['CC2', 'C1'], ['CC2prime', 'CC2'], ['C1prime', 'CC2prime']]
df = get_sterimol_morfeus(df, data_dir=data_dir, sterimol_list=sterimol_list_of_lists, radius=5.5, procs=procs)

#---------------Sterimol DBSTEP-------------------
# Uses DBSTEP to calculate Sterimol L, Bmin, and Bmax values using the
# default grid spacing. This is slower than MORFEUS and not recommended.
# sterimol_list_of_lists = [['C1', 'C4']]
# df = get_sterimol_dbstep(df, data_dir=data_dir, sterimol_list=sterimol_list_of_lists, procs=procs)

#---------------Plane Angle-----------------------
# Plane angle between 2 planes (each defined by 3 atoms)
# plane_atom_names = [['C1', 'CC2', 'CC2prime'], ['CC2', 'CC2prime', 'C1prime']]
# df = get_plane_angle(df, plane_atom_names=plane_atom_names, data_dir=data_dir, procs=procs)

#---------------Time------------------------------
# Extracts the wall time in hours for the entire calculation
df = get_wall_time(df, data_dir=data_dir, procs=procs)

# Save the individual properties for record keeping
df.to_csv(results_dir / 'individual_properties.csv', encoding='utf-8-sig', index=False)
logger.info('Property extraction completed in %.2f seconds', time.time() - t1)
display(df)


# ## Non-parallelized Functions
# These properties are less commonly extracted and are not currently parallelized. Their function in the new notebook is not validated as of March 13th, 2026.

# In[ ]:


#---------------IR--------------------------------
#CAUTION: CANNOT ACCURATELY IDENTIFY ATOM STRETCHES IN MOST CASES (strugges if there is more than one stretch in the defined range)
#IR frequencies and intensities in a specific range (for specific atoms)
#requires the Gaussian keyword = "freq=noraman" in the .com file
#inputs: dataframe, atom1, atom2, frequency_min, frequency_max, intensity_min, intensity_max, threshold
#if you want to collect multiple IR frequencies, you will need to copy/paste this function for each stretch
#we recommend a threshold of 0.0 (may have to adjust)
#df = gp.get_IR(df, "C1", "O2", 1700, 1900, 100, 800, 0.0)

#---------------Sterimol2Vec----------------------
# Sses DBSTEP to calculate Sterimol Bmin and Bmax values at intervals from 0 to end_radius, with a given step_size
# Default grid point spacing (0.05 Angstrom) is used (can use custom spacing or vdw radii in the get_properties_functions script)
#more info here: https://github.com/patonlab/DBSTEP
#inputs: dataframe, list of atom pairs, end_radius, and step_size
#sterimol2vec_list_of_lists = [["C1", "C4"]]
#df = gp.get_sterimol2vec(df, sterimol2vec_list_of_lists, 1, 1.0)


# ## Save collected properties to Excel
# It can be helpful to save the properties of the individual files to read in later if the notebook crashes or if you want to add more properties before post-processsing.

# In[ ]:


# Save to an excel file, ignore index since it is an arbitrary integer index at this point
df.to_excel(results_dir / 'individual_properties.xlsx', engine='openpyxl', index=False)


# ## Post processing
# The following cells perform post-processing during which we compute derivative properties like _min, _max, _boltz, _stddev, and _range.

# In[ ]:


# (Optional) Read in the conformer specific properties. Otherwise, use the df we just created
df = pd.read_excel(results_dir / 'individual_properties.xlsx', header=0, sheet_name='Sheet1')

# Define the character(s) that separates the file name and the conformer number
# e.g., molecule14_5 -> ['molecule14', 5]
separator = '_'

# Drop anything that's not a property we want to average
# Recommended to drop the atom mappings
columns_to_drop = all_mapped_atoms.copy()

# Energy column header for boltzmann weighting and lowest energy conformer
energy_col_header = 'G(T)_spc(Hartree)'

# Specify energy cutoff in kcal/mol to remove conformers above this value before post-processing
energy_cutoff = 5.0

# Get prefix (molecule name) and suffix (conformer number) of the compound then sort the columns.
# Note that if you have delimiters in your compound name (prefix), you must specify multiple
# return keys as a list of integers. See test_2 for an example.
new_cols = pd.DataFrame({'PREFIX': df[FILE_COLUMN_NAME].apply(split_compound_name, delimiter=separator, return_key=0),
                         'SUFFIX': df[FILE_COLUMN_NAME].apply(split_compound_name, delimiter=separator, return_key=1)})

# Do not edit below this line
# Concat the dataframes
df = pd.concat([df, new_cols], axis=1)

# Reorder to put prefix and suffix in the front
df = df[[FILE_COLUMN_NAME, 'PREFIX', 'SUFFIX'] + [x for x in df.columns if x not in [FILE_COLUMN_NAME, 'PREFIX', 'SUFFIX']]]
df.sort_values(by=['PREFIX', 'SUFFIX'], inplace=True)

try:
    df.drop(columns=columns_to_drop, inplace=True)
except KeyError as e:
    logger.error('Could not drop %s from DataFrame', columns_to_drop)

display(df)

# Make a dict of compound_id : number_of_conformer
compound_id_conf_number_dict = {}
for compound_name in set(df['PREFIX'].to_list()):
    compound_id_conf_number_dict[compound_name] = len(df[df['PREFIX'] == compound_name])


# ## Removing columns and rows with missing values
# 
# This cell removes columns for which all values are missing. It then removes rows with any missing values.

# In[ ]:


# Get the columns with all np.nan
missing_value_cols = df.columns[df.isna().all()].tolist()
if missing_value_cols:
    logger.warning('All of the values in some columns were missing\t%s', missing_value_cols)
df.drop(columns=missing_value_cols, inplace=True)

# Drop any row with missing data
to_remove = df[df.isna().any(axis=1)]
to_keep = df[~df.isna().any(axis=1)]

# Display DataFrames
logger.info('There were %d rows removed because of missing values', to_remove.shape[0])
if not to_remove.empty:
    display(to_remove)

# Make a new DataFrame and convert everything but known strings to float
df = to_keep.copy(deep=True)
df[[x for x in df.columns if x not in [FILE_COLUMN_NAME, 'PREFIX', 'SUFFIX', 'file_name', 'file']]] = df[[x for x in df.columns if x not in [FILE_COLUMN_NAME, 'PREFIX', 'SUFFIX', 'file_name']]].astype(float)

logger.info('There are %d rows that remain.', to_keep.shape[0])


# ## Post-processing to generate summary properties
# This cell weights the values according to the Boltzmann distribution. You should change the `temperature`  variable to
# reflect the desired temperature of Boltzmann weights. Modify the `min_property_value` variable to change the property
# with which the XXX_min_conf properties (where XXX is some property like Vbur) are calculated. Set the variable with 
# `min_property_value = None` to skip computing these properties.
# 
# The `ignore_columns` variable is a list of properties that will be skipped when computing summary properties since 
# these properties do not make sense (e.g., H_spc(Hartree)_Boltz).

# In[ ]:


# Set the Boltzmann-weighting temperature
temperature = 298.15

# Set this value to a property name to add properties for the
# conformer with the minimum value of that property (e.g., %Vbur_C3_3.0Å)
# for the "Vbur_min_conf" properties. Set the value to None to skip this.
min_property_value = None

# Define columns that are not processed for summary properties
ignore_columns = [FILE_COLUMN_NAME,
                  energy_col_header,
                  'PREFIX',
                  'SUFFIX',
                  'G(Hartree)',
                  '∆G(Hartree)',
                  '∆G(kcal/mol)',
                  'G(T)_spc(Hartree)',
                  'e^(-∆G/RT)',
                  'Mole Fraction',
                  'wall_time (h)',
                  'E_spc (Hartree)',
                  'H_spc(Hartree)',
                  'T',
                  'T*S',
                  'T*qh_S',
                  'ZPE(Hartree)',
                  'qh_G(T)_spc(Hartree)',
                  ]

ignore_columns.extend(columns_to_drop)

# List of dataframes to concat after everything is done
results = []

# Iterate over all compounds
for compound_identifier, number_of_confs in compound_id_conf_number_dict.items():

    logger.info('Working on %s (nconfs = %d)', compound_identifier, number_of_confs)

    # Read in the PREFIX (which unifies the conformers of a single molecule) as its respective type
    valuesdf = df[df['PREFIX'].astype(str) == str(compound_identifier)].copy(deep=True)

    if len(valuesdf) == 0:
        logger.error('No conformer data was found for %s. Skipping.', compound_identifier)
        continue
    elif len(valuesdf) != number_of_confs:
        logger.warning('Mismatch in number of confs for %s. Should be %d but found %d.', str(compound_identifier), number_of_confs, len(valuesdf))

    # Check for missing data (likely redundant)
    missing_values_check = valuesdf.copy(deep=True)
    valuesdf.dropna(axis=0, how='any', inplace=True)
    if valuesdf.shape[0] != missing_values_check.shape[0]:
        logger.warning('Some rows in DataFrame had missing properties')

    # Get the columns that we want to min/max/average
    columns_to_process = [x for x in valuesdf.columns if x not in ignore_columns]

    # This section starts the summary properties calculations (Boltz, min, max, etc.)
    # Get the relative energy in Hartrees and kcal
    valuesdf["∆G(Hartree)"] = valuesdf[energy_col_header].astype(float) - valuesdf[energy_col_header].astype(float).min()
    valuesdf["∆G(kcal/mol)"] = valuesdf["∆G(Hartree)"] * 627.509

    # Compute the Boltzmann weights (R is in cal/(K*mol))
    valuesdf["e^(-∆G/RT)"] = (valuesdf["∆G(kcal/mol)"] * -1000) / (1.987204 * temperature)
    valuesdf['e^(-∆G/RT)'] = valuesdf['e^(-∆G/RT)'].astype(float).apply(np.exp)
    valuesdf["Mole Fraction"] = valuesdf["e^(-∆G/RT)"] / valuesdf["e^(-∆G/RT)"].sum()

    # Apply the energy cutoff and reset the arbitrary index
    valuesdf.drop(valuesdf[valuesdf["∆G(kcal/mol)"] >= energy_cutoff].index, inplace=True)
    valuesdf = valuesdf.reset_index(drop=True)

    # Get the LEC and property_minimum_conformer as a DataFrame
    lec_df = valuesdf[valuesdf[energy_col_header].astype(float) == valuesdf[energy_col_header].astype(float).min()].copy()
    if min_property_value is not None:
        prop_min = valuesdf[valuesdf[min_property_value].astype(float) == valuesdf[min_property_value].astype(float).min()].copy()

    # Make a dict for holding summary properties
    summary_properties = {}
    summary_properties[FILE_COLUMN_NAME] = compound_identifier

    # Iterate through the columns that have been filtered down
    for column in columns_to_process:

        # Get min max range
        summary_properties[f'{column}_min'] = valuesdf[column].astype(float).min()
        summary_properties[f'{column}_max'] = valuesdf[column].astype(float).max()
        summary_properties[f'{column}_range'] = valuesdf[column].astype(float).max() - valuesdf[column].astype(float).min()

        # Get boltz
        summary_properties[f'{column}_Boltz'] = (valuesdf[column].astype(float) * valuesdf["Mole Fraction"].astype(float)).sum()

        # Get weighted stdev
        # https://www.statology.org/weighted-standard-deviation-excel/

        if len(valuesdf) == 1:
            summary_properties[f'{column}_Boltz_stdev'] = 0
        else:
            boltz = (valuesdf[column] * valuesdf["Mole Fraction"]).sum()
            delta_values_sq = []

            # Get a list of the weights
            w = valuesdf["Mole Fraction"].astype(float).to_numpy()
            delta_values_sq = valuesdf[column].astype(float).to_numpy()
            delta_values_sq = np.array([(x - boltz)**2 for x in delta_values_sq])

            wstdev = np.sqrt( (np.average(delta_values_sq, weights=w)) / (((len(w) - 1) / len(w)) * np.sum(w)) )
            summary_properties[f'{column}_Boltz_stdev'] = wstdev

        # If there are more than one LECs based on the energy
        if len(lec_df) != 1:

            # Check if all of the values are the same for the property we are looking at
            if not all([x == lec_df[column].astype(float).to_list()[0] for x in lec_df[column].astype(float).to_list()]):
                _lowest_energy_debug = valuesdf[energy_col_header].astype(float).min()
                logger.warning('Multiple LECs of the same energy (%.4f %s) were found.', round(_lowest_energy_debug, 5), energy_col_header)
                logger.warning('These identical energy LECs have different values for %s. Taking the average.', lec_df[column].astype(float).to_list())

            # Take the average either way since it's a low cost operation
            summary_properties[f'{column}_low_E'] = np.average(lec_df[column].astype(float).to_list())
        else:
            # Take the first and only value
            summary_properties[f'{column}_low_E'] = lec_df[column].astype(float).values[0]

        # If there are more than one min_confs based on the property
        if min_property_value is not None:
            if len(prop_min) != 1:

                # Check if all of the values are the same for the property we are looking at
                if not all([x == prop_min[column].astype(float).to_list()[0] for x in prop_min[column].astype(float).to_list()]):
                    _prop_min_debug = valuesdf[prop_min].astype(float).min()
                    logger.warning('Multiple conformers of the same minimum %s (%.4f %s) were found.', prop_min, round(_prop_min_debug, 5), energy_col_header)
                    logger.warning('These identical energy %s have different values for %s. Taking the average.', prop_min, lec_df[column].astype(float).to_list())

                # Take the average either way since it's a low cost operation
                summary_properties[f'{column}_({min_property_value}_min_conf)'] = np.average(lec_df[column].astype(float).to_list())
            else:
                # Take the first and only value
                summary_properties[f'{column}_({min_property_value}_min_conf)'] = lec_df[column].astype(float).values[0]

    results.append(pd.Series(summary_properties))

summary_properties = pd.DataFrame(results)
display(summary_properties)


# ## Save the Results

# In[ ]:


summary_properties.to_csv(results_dir / f'summary_properties.csv', encoding='utf-8-sig', index=False)
summary_properties.to_excel(results_dir / f'summary_properties.xlsx', engine='openpyxl', index=False)


# ## Making Derivative Properties
# Because some molecules have symmetric substructures that make atom assignment challenging, it can be advantageous to "derivatize" the summary properties. For instance, the two alpha carbons of a ketone may be important for a particular substitution reaction, but it is difficult to assign descriptors to a specific alpha carbon across a series of ketones. 
# 
# <img src="./data/figure1.png" style="display:block; margin:auto; width:500px;">
# 
# Thus, we can perform mathematical operations on the descriptors to capture substructure symmetry. The alpha carbon descriptors can be further processed to describe the __average__, __min__, or __max__ value of a property across both alpha carbons. Further derivitization like __delta__ could be used to capture the difference in a property between equivalent atoms. 
# 
# To further exemplify this issue, consider the pyridine example above. The yield of a reaction that occurs at the C1/C5 
# position (next to nitrogen) might correlate with positively with an increasing positive charge on C1/C5. However, the
# greatest partial positive charge is not guarenteed to be on exclusively C1 or C5 – it can switch. However, the 
# __maximum__ value between the charge on C1 or C5 will always be the one with the greatest positive charge. That 
# derivative descriptor (Ex. charge_C1_or_C5_Boltz_max) will ...
#  

# In[ ]:


# Read in the summary properties
df = pd.read_csv(results_dir / f'summary_properties.csv', index_col=0)

#raise NotImplementedError('This section requires heavy edits depending on your project and should be used cautiously.')

# Get a list of descriptors that are candidates for derivative properties
descriptors = [x for x in df.columns if x not in ignore_columns]

# Summary property suffixes that will be considered. This can get confusing
# very quickly, so we'll stick with _Boltz descriptors for now.
suffixes = ['Boltz']

descriptors_to_symmetrize = [
    # C1/C1prime single atom properties
    ['NBO_charge_C1', 'NBO_charge_C1prime'],
    ['ChelpG_charge_C1', 'ChelpG_charge_C1prime'],
    ['Hirsh_CM5_charge_C1', 'Hirsh_CM5_charge_C1prime'],
    ['Hirsh_atom_dipole_C1', 'Hirsh_atom_dipole_C1prime'],
    ['NMR_shift_C1', 'NMR_shift_C1prime'],

    # CC2/CC2prime single atom properties
    ['NBO_charge_CC2', 'NBO_charge_CC2prime'],
    ['ChelpG_charge_CC2', 'ChelpG_charge_CC2prime'],
    ['Hirsh_CM5_charge_CC2', 'Hirsh_CM5_charge_CC2prime'],
    ['Hirsh_atom_dipole_CC2', 'Hirsh_atom_dipole_CC2prime'],
    ['NMR_shift_CC2', 'NMR_shift_CC2prime'],

    # Distances (this didnt work for some reason for the derivatizing)
    # ['distance_C1_CC2(Å)', 'distance_CC2prime_C1prime(Å)'],

    # Vbur
    ['%Vbur_C1_2.0Å', '%Vbur_C1prime_2.0Å'],
    ['%Vbur_C1_2.5Å', '%Vbur_C1prime_2.5Å'],
    ['%Vbur_C1_3.0Å', '%Vbur_C1prime_3.0Å'],
    ['%Vbur_C1_3.5Å', '%Vbur_C1prime_3.5Å'],
    ['%Vbur_C1_4.0Å', '%Vbur_C1prime_4.0Å'],
    ['%Vbur_C1_4.5Å', '%Vbur_C1prime_4.5Å'],

    ['%Vbur_CC2_2.0Å', '%Vbur_CC2prime_2.0Å'],
    ['%Vbur_CC2_2.5Å', '%Vbur_CC2prime_2.5Å'],
    ['%Vbur_CC2_3.0Å', '%Vbur_CC2prime_3.0Å'],
    ['%Vbur_CC2_3.5Å', '%Vbur_CC2prime_3.5Å'],
    ['%Vbur_CC2_4.0Å', '%Vbur_CC2prime_4.0Å'],
    ['%Vbur_CC2_4.5Å', '%Vbur_CC2prime_4.5Å'],

]
# Keep a list of descriptors that we will drop because they've been symmetrized
# to_drop = []

# Accumulate new columns here to avoid repeated dataframe fragmentation
new_columns = {}

for descriptor_group in descriptors_to_symmetrize:
    for s in suffixes:

        # Get all the descriptors with the right suffix
        to_sym = [x for x in descriptors if x.endswith(f'_{s}') and any([z in x for z in descriptor_group])]

        if to_sym == []:
            continue

        # Symmetrize the name
        new_descriptor_base = [x.replace('C1prime', 'C1') for x in to_sym]
        new_descriptor_base = [x.replace('C2prime', 'C2') for x in new_descriptor_base]
        new_descriptor_base = list(set(new_descriptor_base))

        if len(new_descriptor_base) != 1:
            raise ValueError(new_descriptor_base, len(new_descriptor_base))

        new_descriptor_base = new_descriptor_base[0]

        to_sym_str = f'{to_sym!s:<100.100}'
        print(f'Symmetrizing {to_sym_str}\tNew base name is: {new_descriptor_base}')

        sym_df = df[to_sym]
        new_columns[f'{new_descriptor_base}_minsym'] = sym_df.min(axis=1)
        new_columns[f'{new_descriptor_base}_maxsym'] = sym_df.max(axis=1)
        new_columns[f'{new_descriptor_base}_averagesym'] = sym_df.mean(axis=1)
        if len(to_sym) == 2:
            new_columns[f'{new_descriptor_base}_deltasym'] = abs(df[to_sym[0]] - df[to_sym[1]])
        else:
            new_columns[f'{new_descriptor_base}_rangesym'] = abs(sym_df.max(axis=1) - sym_df.min(axis=1))

        # to_drop.extend(to_sym)

# df.drop(columns=to_drop, inplace=True)

# Single concat — one memory allocation for all new columns
df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

df.to_csv(results_dir / 'summary_and_derivative_properties.csv', encoding='utf-8-sig')

display(df)

