import numpy as np
import pandas as pd
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
        concentration_list = np.zeros((iterations, len(self.nodes)))
        for i in range(iterations):
            for j,node in enumerate(self.nodes):
                concentration_list[i][j] = node.concentration
                pollutant_flow = node.people * self.HUMAN_EXHALATION_FLOW
                for conn in self.connections:
                    if conn.node1 == node:
                        if conn.flow <= 0:
                            pollutant_flow += conn.flow * (node.concentration - conn.node2.concentration if conn.node2 else node.concentration - conn.outside_concentration)
                    elif conn.node2 == node:
                        if conn.flow >= 0:
                            pollutant_flow -= conn.flow * (node.concentration - conn.node2.concentration)
                node.update_concentration((pollutant_flow/node.size)*delta_times + node.concentration)
        return concentration_list

    def _random_people_number(self):
        """
        Generate a list of random people number for each node.
        
        Args:
            lenght (int): The number of nodes.
            sum_people (int): The total number of people.
        
        Returns:
            list: A list of random people number for each node.
        """
        lenght = len(self.nodes)
        sum_people = sum([node.people for node in self.nodes])
        people_list = np.zeros(lenght)
        while np.sum(people_list) < sum_people:
            people_list[np.random.randint(lenght)] += 1
        
        for i,node in enumerate(self.nodes):
            node.update_people(int(people_list[i]))

    def flow_balance(self):
        """
        Balances the flow in the ventilation network by solving a non-linear equation system.

        Returns:
            list: The solution to the non-linear equation system.
        """
        network = self.VentilationNetwork
        pressure_list = [0] * len(network.nodes)
        result = root(self.flow_equation, pressure_list) # 使用root函数求解非线性方程组
        if not result.success:
            raise ValueError("Flow balance calculation failed.")
        return result.x
    
    def _output_result(self,result:dict):
        """
        Output the result as datasets.
        """
        data = []
        keys = ['pressure', 'concentration', 'people']
        for key in result:
            data.extend([{'Node': key, **dict(zip(keys, values))} for values in zip(result[key]['pressure'],result[key]['concentration'],result[key]['people'])])
        df = pd.DataFrame(data)
        return df
    
    def main(self,output_path:str=None):
        """
        Main function to run the calculation.
        """
        self.flow_balance()
        result = {f'Node {node.identifier}': {'pressure':[node.pressure],'concentration':[node.concentration],'people':[node.people]} for node in self.nodes}
        for _ in range(3):
            concentration_list = self.concentration_calculation()
            for i,node in enumerate(self.nodes):
                result[f'Node {node.identifier}']['concentration'].extend(concentration_list[:,i])
                self._random_people_number()
                result[f'Node {node.identifier}']['people'].extend((np.ones_like(concentration_list[:,i])*node.people).astype(int))
                result[f'Node {node.identifier}']['pressure'].extend([node.pressure]*len(concentration_list[:,i]))
        if output_path:
            df = self._output_result(result)
            df.to_csv(output_path,index=False)

if __name__ == '__main__':
    network = VentilationNetwork(5)
    caculation = Caculation(network)
    caculation.main()