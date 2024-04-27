import pandas as pd
import numpy as np


def prep_matrix_data(file_path: str) -> tuple:
    """
    Prepares matrix data for training.

    Args:
        file_path (str): File path to the matrix data.

    Returns:
        tuple: A tuple containing the building identifier and the matrix data.
    """
    value = pd.read_csv(file_path).values
    building_identifier = file_path.stem.replace("_matrix", "")
    return building_identifier, value


def prep_non_matrix_data(file_path: str) -> tuple:
    """
    Preprocesses non-matrix data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the building identifier, values, and labels.
            - building_identifier (str): The identifier of the building.
            - values (ndarray): An array of shape (num_nodes, num_times-1, 2) containing the concentration and people values.
            - labels (ndarray): An array of shape (num_nodes, 1) containing the concentration labels.
    """
    data = pd.read_csv(file_path).loc[:, [
        'Node', 'time', 'concentration', 'people']]
    data.replace('Node ', '', inplace=True, regex=True)
    times = data['time'].unique()
    nodes = data['Node'].unique()
    values = np.zeros((len(nodes), len(times)-1, 2))
    labels = np.zeros((len(nodes), 1))
    for i, node in enumerate(nodes):
        node_data = data[data['Node'] == node]
        concentration = node_data['concentration'].values
        people = node_data['people'].values
        values[i, :, 0] = concentration[:-1]
        values[i, :, 1] = people[:-1]
        labels[i] = concentration[-1]
    building_identifier = file_path.stem
    return building_identifier, values, labels


def prep_data(file_path: str) -> tuple:
    """
    Prepare the data based on the file path.

    Args:
        file_path (str): The path of the file.

    Returns:
        tuple: A tuple containing the prepared data.
    """
    if "_matrix" in file_path.name:
        return prep_matrix_data(file_path)
    else:
        return prep_non_matrix_data(file_path)
