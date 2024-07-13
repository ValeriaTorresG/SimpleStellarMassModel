#!/usr/bin/env python
import argparse
import logging
from astropy.io import fits
from astropy.table import Table as t
import astropy.units as u
import h5py
import numpy as np
import csv
import time as t


def read_files(fits_path, bgs_path):
    """
    Read data from FITS and HDF5 files and extract unique rosette numbers.
    This function opens a FITS file and an HDF5 file from the provided paths,
    extracts data, and computes the unique rosette numbers from the FITS data.

    Parameters:
    - fits_path (str): The file path to the BGS clustering FITS file.
    - bgs_path (str): The file path to the PROVABGS HDF5 file.

    Returns:
    Returns:
    - tuple: A tuple containing:
        1. An array of unique rosette numbers extracted from the FITS file.
        2. The dataset from the HDF5 file.
        3. The data from the FITS file.

    Raises:
    - FileNotFoundError: If either the FITS file or the HDF5 file does not exist
      at the specified path.
    - IOError: If there is an error reading from the FITS or HDF5 file.

    Note:
    - The function assumes the FITS file contains a table at extension 1 with a column
    named 'ROSETTE_NUMBER'.
    - The HDF5 file must contain a dataset in '__astropy_table__'.
    """
    try:
        hdu = fits.open(f'{fits_path}')
        data_bgs = hdu[1].data
    except FileNotFoundError:
        raise FileNotFoundError(f"The FITS file at {fits_path} was not found.")
    except Exception as e:
        raise IOError(f"An error occurred while opening the FITS file: {str(e)}")

    try:
        data_prova = h5py.File(f'{bgs_path}')
        dataset = data_prova['__astropy_table__']
        n_r = np.unique(data_bgs['ROSETTE_NUMBER'])
        return n_r, dataset, data_bgs
    except FileNotFoundError:
        data_prova.close()
        raise FileNotFoundError(f"The HDF5 file at {bgs_path} was not found.")
    except Exception as e:
        data_prova.close()
        raise IOError(f"An error occurred while opening the HDF5 file: {str(e)}")


def read_data(r, data_bgs, dataset):
    """
    Extracts and filters data for a given rosette number from a provided dataset.
    This function selects entries from a dataset that match a specified rosette number.
    It collects various flux measurements and other properties for targets
    with positive mass and non-negative flux values.

    Parameters:
    - r (int): The rosette number to filter the dataset on.
    - data_bgs (ndarray): Array containing the rosette data.
    - dataset (list of tuples): Dataset containing target information.

    Returns:
    - list: A list of tuples, each containing data for targets in the specified rosette
            that meet the selection criteria (positive mass and non-negative fluxes).
            Each inner tuple contains:
            - TARGETID
            - Stellar mass
            - Flux measurements across five bands (G, R, Z, W1, W2)
            - Redshift (Z)

    Note:
    - The function filters data based on stellar mass and flux measurements being non-negative.
    """
    rosette_n = data_bgs[data_bgs['ROSETTE_NUMBER'] == r]
    ids = rosette_n['TARGETID']
    selected = [(row[0], row[15]) for row in dataset if row[0] in ids]
    mass, rosettes = selected, []
    for j in range(len(mass)):
        data_j = data_bgs[data_bgs['TARGETID'] == mass[j][0]]
        flux_g = data_j['FLUX_G_DERED'][0]
        flux_r = data_j['FLUX_R_DERED'][0]
        flux_z = data_j['FLUX_Z_DERED'][0]
        flux_w1 = data_j['FLUX_W1_DERED'][0]
        flux_w2 = data_j['FLUX_W2_DERED'][0]
        z = data_j['Z'][0]
        if (mass[j][1]>=0) and (flux_w1>=0) and (flux_w2>=0):
            rosettes.append((mass[j][0], mass[j][1], flux_g, flux_r, flux_z, flux_w1, flux_w2, z))
    return rosettes


def write_data(r, rosette):
    """
    Writes data for a specified rosette to a CSV file.
    This function takes a list of data tuples for a specific rosette and writes it to a CSV file.
    Each tuple contains details of a target in the rosette, including target ID, mass, various flux measurements,
    and redshift, which are written to a CSV file named according to the rosette number.

    Parameters:
    - r (int): The rosette number, which is used to name the output file.
    - rosette (list of tuples): A list of tuples where each tuple contains the data for one target.
      Each tuple should have the following format:
      (target_id, mass, flux_g, flux_r, flux_z, flux_w1, flux_w2, z)

    Output:
    - A new CSV file named 'rosette{r}.csv' located in the '../data/' directory.
      Each row corresponds to a target's data.

    Note:
    - The function assumes that the '../data/' directory exists and is writable.
    """
    with open(f'../data/rosette{r}.csv', mode='w', newline='') as csv_file:
        fieldnames = ['TARGET_ID', 'PROVABGS_LOGMSTAR_BF', 'FLUX_G_DERED', 'FLUX_R_DERED',
                      'FLUX_Z_DERED', 'FLUX_W1_DERED', 'FLUX_W2_DERED', 'Z']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for target_id, mass, flux_g, flux_r, flux_z, flux_w1, flux_w2, z in rosette:
            writer.writerow({
                'TARGET_ID': target_id,
                'PROVABGS_LOGMSTAR_BF': mass,
                'FLUX_G_DERED': flux_g,
                'FLUX_R_DERED': flux_r,
                'FLUX_Z_DERED': flux_z,
                'FLUX_W1_DERED': flux_w1,
                'FLUX_W2_DERED': flux_w2,
                'Z': z
            })


def main():
    """
    Main function to process data from FITS and HDF5 files.
    This function parses command-line arguments for file paths, reads the necessary data,
    and processes and writes information for each rosette, logging the time taken
    for each operation.

    Command-Line Arguments:
    --fits_path: Path to the FITS file.
    --bgs_path: Path to the HDF5 file.

    Each rosette's data is processed to create a CSV file and the process time and number
    of galaxies processed are printed to the console.
    """
    parser = argparse.ArgumentParser(description='rosettes_paths')
    parser.add_argument('--fits_path', type=str, required=True)
    parser.add_argument('--bgs_path', type=str, required=True)
    args = parser.parse_args()
    fits_path, bgs_path = args.fits_path, args.bgs_path
    rosettes, dataset, data_bgs = read_files(fits_path, bgs_path)

    logging.basicConfig(
    filename='../job_management/logs/write_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

    for r in rosettes:
        init = t.time()
        r_info = read_data(r, data_bgs, dataset)
        write_data(r, r_info)
        end = t.time()
        logging.info(f'Rosette{r} csv: {round((end-init)/60,2)} min, {len(r_info)} galaxies')
        print(f'time writing rosette{r} csv: {round((end-init)/60,2)} min, {len(r_info)} galaxies')


if __name__ == "__main__":
    main()