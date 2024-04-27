from src.ventilation.build_data import multi_step_data
from src.deepl.train import *

if __name__ == '__main__':
    # multi_step_data(step=10, output=True)
    Train().train()
    pass