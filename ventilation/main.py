from network import VentilationNetwork
from caculation import *


if __name__ == '__main__':
    network = VentilationNetwork(5)
    caculation = Caculation(network)
    caculation.main(output_path='output.csv')