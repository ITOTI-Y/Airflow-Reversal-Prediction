import numpy as np
from typing import List
from node import Node
from connection import Connection
from network import VentilationNetwork
from scipy.optimize import fsolve, root, differential_evolution


class Caculation:
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
        self.nodes = VentilationNetwork.nodes
        self.connections = VentilationNetwork.connections
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
        delat_time = 60 # Time step in seconds
        for node in self.nodes:
            concentration_change = 0
            for conn in self.connections:
                if conn.node1 == node:
                    concentration_change -= conn.flow * (node.co2 - conn.node2.co2 if conn.node2 else conn.outside_concentration)
                elif conn.node2 == node:
                    concentration_change += conn.flow * (node.co2 - conn.node1.co2)



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
        network.visualize_network()
        return result.x

if __name__ == '__main__':
    network = VentilationNetwork(1)
    # network.visualize_network()
    caculation = Caculation(network)
    caculation.flow_balance()
    print(network.nodes)