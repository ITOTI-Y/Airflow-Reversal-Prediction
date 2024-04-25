import pandas as pd


def prep_data(file_path: str):
    data = pd.read_csv(file_path).loc[:, [
        'Node', 'time', 'concentration', 'people']]
    building_identifier = file_path.stem
    data['building'] = building_identifier
    return data
