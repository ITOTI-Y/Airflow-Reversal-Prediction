from src.ventilation.build_data import *
from src.deepl.train import *

if __name__ == '__main__':
    # multi_step_data(step=3, output=True, show=False)
    Train().train()
    pass