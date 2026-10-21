#!/usr/bin/env python3
# coding: utf-8

'''
'''

import time
import itertools
import multiprocessing

import pandas as pd
from tqdm import tqdm

def normalize_name(x):
    """Convert input to string and remove spaces for consistent matching."""
    return str(x).strip()

def process_difference_props(m1_m2_tuple: tuple[str],
                             dataframe: pd.DataFrame,
                             properties: list[str]):

    m1 = m1_m2_tuple[0]
    m2 = m1_m2_tuple[1]

    # Get the sub dataframes for these molecules
    _m1_df = dataframe[dataframe['Compound_Name'].astype(str) == m1]
    _m2_df = dataframe[dataframe['Compound_Name'].astype(str) == m2]

    # Sanity checks
    if _m1_df.shape[0] != 1:
        raise ValueError(f'Expected 1 row and found {_m1_df.shape[0]}')
    if _m2_df.shape[0] != 1:
        raise ValueError(f'Expected 1 row and found {_m2_df.shape[0]}')

    results = {'M1': m1,
               'M2': m2}

    # Convert each property to a new name containing the prefix based on M1 or M2
    # i.e., Vbur_2.0_A -> M1_Vbur_2.0_A
    for prop in properties:

        # Get the M1/M2 values (this is only valid because we know there is one value in this pd.Series)
        m1_val = _m1_df[prop].astype(float).values[0]
        m2_val = _m2_df[prop].astype(float).values[0]

        # Add those to the row of the experimental “row”
        results[f'M1_{prop}'] = m1_val
        results[f'M2_{prop}'] = m2_val

        if (prop.endswith('_Boltz')) or (prop.endswith('_low_E')) or (prop.endswith('Boltz_averagesym')) or (prop.endswith('Boltz_deltasym')):
            _prop_1 = f'M1_{prop}'
            _prop_2 = f'M2_{prop}'
            new_prop_name = f'{_prop_1}-{_prop_2}'
            results[new_prop_name] = m1_val - m2_val

    return pd.Series(results)


def main():
    # Read in the computational data
    comp = pd.read_excel('summary_and_derivative_properties_all850monomers_intuitivelyedited_with_smiles_and_duplicates_manuallyeliminateduplicates.xlsx', sheet_name='Sheet4', engine='openpyxl')

    # Read in the experimental data
    exp_original = pd.read_excel(
        '225experimentpairs.xlsx',
        sheet_name='Sheet1',
        engine='openpyxl'
    )

    # For debugging
    # exp_original = exp_original.head(500)

    # Make a copy of the experimental df in case we need it again
    exp = exp_original.copy(deep=True)

    # --- NORMALIZE COMPOUND NAMES ---
    comp['Compound_Name'] = comp['Compound_Name'].apply(normalize_name)
    exp['M1'] = exp['M1'].apply(normalize_name)
    exp['M2'] = exp['M2'].apply(normalize_name)

    # Get a list of properties that we want to copy over
    properties = list(comp.columns.astype(str))

    # Remove specific properties
    properties = [x for x in properties if x not in ['Compound_Name']]

    # Build the input tuples once
    t1 = time.time()
    experiments_to_process = [(x.get('M1'), x.get('M2')) for _, x in exp.iterrows()]
    args = list(
        zip(
            experiments_to_process,
            itertools.repeat(comp),
            itertools.repeat(properties)
        )
    )

    comp_names = set(comp['Compound_Name'])
    missing_m1 = set(exp['M1']) - comp_names
    missing_m2 = set(exp['M2']) - comp_names

    if missing_m1 or missing_m2:
        print("⚠️ WARNING: Some compounds are missing from the computational dataset!")
        if missing_m1:
            print("Missing M1:", missing_m1)
        if missing_m2:
            print("Missing M2:", missing_m2)
        # Optionally: raise an error to stop processing
        # raise ValueError("Missing compounds detected. Fix before proceeding.")

    # Run multiprocessing with a progress bar
    with multiprocessing.Pool() as p:
        results = list(
            tqdm(
                p.starmap(process_difference_props, args),
                total=len(args),
                desc='Processing experiments'
            )
        )

    t2 = time.time()
    print(f'Finished in {round(t2-t1, 2)} seconds')

    df = pd.DataFrame(results)
    df.to_csv('summary_and_derivative_properties_all850monomers_intuitivelyedited_with_smiles_and_duplicates_manuallyeliminateduplicates_225expscombinatorialized.csv', index=False)
    df.to_excel('summary_and_derivative_properties_all850monomers_intuitivelyedited_with_smiles_and_duplicates_manuallyeliminateduplicates_225expscombinatorialized.xlsx', index=False)

if __name__ == "__main__":
    main()