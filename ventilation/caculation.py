import numpy as np
from typing import List
from node import Node
from connection import Connection
from network import VentilationNetwork
from scipy.optimize import fsolve, root, differential_evolution


class Caculation:
    HUMAN_EXHALATION = 0.0001 # 人体潮气量 m^3/s
    HUMAN_EXHALATION_CONCENTRATION = 40000 # 人体潮气二氧化碳浓度 ppm
    HUMAN_EXHALATION_FLOW = HUMAN_EXHALATION * HUMAN_EXHALATION_CONCENTRATION # 人体潮气二氧化碳流量 ppm*m^3/s

    def __init__(self, VentilationNetwork: VentilationNetwork):
        """
        Initializes an instance of the Calculation class.

        Args:
            VentilationNetwork (VentilationNetwork): The VentilationNetwork object to be used for calculations.

        Attributes:
            VentilationNetwork (VentilationNetwork): The VentilationNetwork object to be used for calculations.
            nodes (list): The list of nodes in the VentilationNetwork.
            connections (list): The list of connections in the VentilationNetwork.
        """
        self.VentilationNetwork = VentilationNetwork
        self.nodes:List[Node] = VentilationNetwork.nodes
        self.connections:List[Connection] = VentilationNetwork.connections
    def flow_equation(self, pressure_list: list) -> list:
        """
        Calculates the flow equations for the given pressure list.

        Args:
            pressure_list (list): A list of pressures for each node.

        Returns:
            list: A list of flow equations for each node.
        """
        equations = []
        nodes = self.nodes
        connections = self.connections

        # Update pressure for each node
        for node in nodes:
            node.update_pressure(pressure_list[node.identifier])

        # Update flow for each connection
        for conn in connections:
            conn.update_flow()

        # Calculate flow for each node
        for node in nodes:
            flow = 0
            for conn in connections:
                if conn.node1 == node:
                    flow -= conn.flow
                elif conn.node2 == node:
                    flow += conn.flow
            equations.append(flow)

        return equations
    
    def concentration_calculation(self):
        delta_times = 1
        iterations = 1800
        result_list = np.zeros((iterations, len(self.nodes)))
        for i in range(iterations):
            for j,node in enumerate(self.nodes):
                result_list[i][j] = node.concentration
                pollutant_flow = node.people * self.HUMAN_EXHALATION_FLOW
                for conn in self.connections:
                    if conn.node1 == node:
                        if conn.flow <= 0:
                            pollutant_flow += conn.flow * (node.concentration - conn.node2.concentration if conn.node2 else node.concentration - conn.outside_concentration)
                    elif conn.node2 == node:
                        if conn.flow >= 0:
                            pollutant_flow -= conn.flow * (node.concentration - conn.node2.concentration)
                node.update_concentration((pollutant_flow/node.size)*delta_times + node.concentration)
        return result_list

    def flow_balance(self):
        """
        Balances the flow in the ventilation network by solving a non-linear equation system.

        Returns:
            list: The solution to the non-linear equation system.
        """
        network = self.VentilationNetwork
        pressure_list = [0] * len(network.nodes)
        result = root(self.flow_equation, pressure_list) # 使用root函数求解非线性方程组
        if result.success:
            print('解：', result.x)
        else:
            print('找不到解。')
        return result.x

if __name__ == '__main__':
    network = VentilationNetwork(5)
    caculation = Caculation(network)
    caculation.flow_balance()
    result1 = caculation.concentration_calculation()
    network.visualize_network()
    network.nodes[0].people = 1
    # result2 = caculation.concentration_calculation()
    pass
    # print(network.nodes)