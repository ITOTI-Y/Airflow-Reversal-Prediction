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
    TIME = 1
    data = pd.read_csv(file_path).loc[:, [
        'Node', 'time', 'concentration', 'people', 'size']]
    data.replace('Node ', '', inplace=True, regex=True)
    times = data['time'].unique()
    nodes = data['Node'].unique()
    values = np.zeros((len(nodes), len(times)-TIME, 3))
    labels = np.zeros((len(nodes), 1))
    for i, node in enumerate(nodes):
        node_data = data[data['Node'] == node]
        concentration = node_data['concentration'].values
        people = node_data['people'].values
        size = node_data['size'].values
        values[i, :, 0] = concentration[:-TIME]
        values[i, :, 1] = people[:-TIME]
        values[i, :, 2] = size[:-TIME]
        labels[i] = concentration[-TIME]
    building_identifier = file_path.stem
    return building_identifier, values, labels

def prep_flow_data(file_path: str) -> tuple:
    """
    Preprocesses flow data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the building identifier and values.
            - building_identifier (str): The identifier of the building.
            - values (ndarray): An array of shape (edge_index, 1) containing the connections and flow value.
    """
    data = pd.read_csv(file_path).sort_values(by=['Node1', 'Node2'])
    mask = data['Node1'] > data['Node2']
    data.loc[mask, ['Node1', 'Node2']] = data.loc[mask, ['Node2', 'Node1']].values
    data.loc[mask, 'Flow'] = -data.loc[mask, 'Flow']
    data = data.sort_values(by=['Node1', 'Node2'])
    data = data.reset_index(drop=True)
    building_identifier = file_path.stem.replace("_flow", "")
    return building_identifier, data.to_numpy()


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
    elif "_flow" in file_path.name:
        return prep_flow_data(file_path)
    else:
        return prep_non_matrix_data(file_path)
