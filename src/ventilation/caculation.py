import numpy as np
import pandas as pd
from typing import List
from .node import Node
from .connection import Connection
from .network import VentilationNetwork
from ..config import CALCULATE_CONFIG
from scipy.optimize import root

ENV_CONFIG = CALCULATE_CONFIG()
class Caculation:
    """A class used to perform calculations on a VentilationNetwork.

    Attributes:
        HUMAN_EXHALATION (float): The volume of human exhalation in m^3/s.
        HUMAN_EXHALATION_CONCENTRATION (float): The concentration of CO2 in human exhalation in ppm.
        HUMAN_EXHALATION_FLOW (float): The flow of CO2 in human exhalation in ppm*m^3/s.
        VentilationNetwork (VentilationNetwork): The VentilationNetwork object to be used for calculations.
        nodes (List[Node]): The list of nodes in the VentilationNetwork.
        connections (List[Connection]): The list of connections in the VentilationNetwork.
    """

    def __init__(self, VentilationNetwork: VentilationNetwork):
        """Initializes an instance of the Calculation class.

        Args:
            VentilationNetwork (VentilationNetwork): The VentilationNetwork object to be used for calculations.
        """
        self.VentilationNetwork = VentilationNetwork
        self.nodes: List[Node] = VentilationNetwork.nodes
        self.connections: List[Connection] = VentilationNetwork.connections
        self.start_time = 0
        self.end_time = 0

    def flow_equation(self, pressure_list: List[float]) -> List[float]:
        """Calculates the flow equations for the given pressure list.

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

    def concentration_calculation(self, total_time: int = 1800, time_step: int = 1):
        """Calculates the concentration for each node over a number of iterations.

        Args:
            total_time (int, optional): The total time for the calculation. Defaults to 1800.
            delta_time (int, optional): The time step for the calculation. Defaults to 1.

        Returns:
            tuple: A tuple containing the times and the concentration list.
        """
        time_step = time_step
        iters = int(total_time / time_step)
        concentration_list = np.zeros((iters, len(self.nodes)))
        for i in range(iters):
            for j, node in enumerate(self.nodes):
                concentration_list[i][j] = node.concentration
                pollutant_flow = node.people * ENV_CONFIG.HUMAN_EXHALATION_FLOW
                for conn in self.connections:
                    if conn.node1 == node:
                        if conn.flow <= 0:
                            pollutant_flow += conn.flow * \
                                (node.concentration - conn.node2.concentration if conn.node2 else node.concentration -
                                 conn.outside_concentration)
                    elif conn.node2 == node:
                        if conn.flow >= 0:
                            pollutant_flow -= conn.flow * \
                                (node.concentration - conn.node2.concentration)
                node.update_concentration(
                    (pollutant_flow/node.size)*time_step + node.concentration)
        self.end_time += total_time
        times = np.arange(self.start_time, self.end_time, time_step).reshape(-1, 1)
        self.start_time += total_time
        outside_concentarion_list = np.ones_like(times) * ENV_CONFIG.OUTSIDE_CONCENTRATION
        return times, np.concatenate([concentration_list,outside_concentarion_list],axis=1)

    def flow_balance(self):
        """Balances the flow in the ventilation network by solving a non-linear equation system.

        Returns:
            list: The solution to the non-linear equation system.

        Raises:
            ValueError: If the flow balance calculation fails.
        """
        network = self.VentilationNetwork
        pressure_list = [0] * len(network.nodes)
        result = root(self.flow_equation, pressure_list)  # 使用root函数求解非线性方程组
        if not result.success:
            raise ValueError("Flow balance calculation failed.")
        return result.x

    def output_result(self, result: dict):
        """Outputs the result as a DataFrame.

        Args:
            result (dict): The result to be output.

        Returns:
            DataFrame: The result as a DataFrame.
        """
        data = []
        params = list(next(iter(result.values())).keys())
        for key in result:
            param_values = [result[key][param] for param in params]
            data.extend([{'Node': key, **dict(zip(params, param_value))}
                        for param_value in zip(*param_values)])  # *用作列表/元组解包，**用作字典解包
        df = pd.DataFrame(data)
        return df


def main(network: VentilationNetwork, caculation: Caculation, seed: int = 0, output: bool = False):
    """Main function to run the calculation.

    This function runs the flow balance calculation, performs concentration calculations,
    and optionally outputs the results to a CSV file.

    Args:
        network (VentilationNetwork): The VentilationNetwork to be used.
        caculation (Caculation): The Caculation object to be used.
        seed (int, optional): The seed for random number generation. Defaults to 0.
        output (bool, optional): Whether to output the results to a CSV file. Defaults to False.

    """
    caculation.flow_balance()
    result = {f'Node {node.identifier}': {'time': [], 'pressure': [],
                                          'concentration': [], 'people': [], 'size': []} for node in caculation.nodes}
    for _ in range(3):
        times, concentration_list = caculation.concentration_calculation()
        for i, node in enumerate(caculation.nodes):
            result[f'Node {node.identifier}']['time'].extend(times[:, 0])
            result[f'Node {node.identifier}']['concentration'].extend(
                concentration_list[:, i])
            result[f'Node {node.identifier}']['people'].extend(
                (np.ones_like(concentration_list[:, i])*node.people).astype(int))
            result[f'Node {node.identifier}']['pressure'].extend(
                [node.pressure]*len(concentration_list[:, i]))
            result[f'Node {node.identifier}']['size'].extend(
                [node.size]*len(concentration_list[:, i]))
            network.random_people_number()
    if output:
        df = caculation.output_result(result)
        df.to_csv(
            f'./ventilation/data/N{len(network.nodes)}_C{len(network.connections)}.csv', index=False)
    network.visualize_network()


if __name__ == '__main__':
    network = VentilationNetwork(5)
    caculation = Caculation(network)
    main(network, caculation)
