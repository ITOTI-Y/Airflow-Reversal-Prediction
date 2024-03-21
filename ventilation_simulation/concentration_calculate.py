import numpy as np
from network import Network


class Concentration_calculate:
    def __init__(self,calcultated_network,total_people=10):
        self.network = calcultated_network
        self.nodes = self.network.nodes
        self.connections = self.network.connections
        self.total_people = total_people
        self.delta_t = 1
        self.external_ppm = 440

    def random_people(self):
        total_people = self.total_people
        for i in range(total_people):
            node = np.random.choice(self.nodes)
            node.update_people(node.people + 1)

    def calculate_concentration(self,iterations=5000,tolerance=1e-6):
        pass

if __name__ == "__main__":
    network = Network()
    network = network.build_random_network(5,2,1)
    network.calculate_network()
    result = Concentration_calculate(network)
    result.random_people()
    # result.calculate_concentration()
    network.display()