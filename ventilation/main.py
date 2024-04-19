from network import VentilationNetwork
from caculation import *


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
                                          'concentration': [], 'people': []} for node in caculation.nodes}
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
            network.random_people_number()
    if output:
        df = caculation.output_result(result)
        outside_node_num = sum([1 for conn in network.connections if conn.node2 == None])
        df.to_csv(
            f'./ventilation/data/N{len(network.nodes)}_C{len(network.connections)}_O{outside_node_num}.csv', index=False)
    network.visualize_network()


if __name__ == '__main__':
    """Main entry point of the script.

    This block of code is executed if the script is run directly. It creates a VentilationNetwork,
    a Caculation object, and runs the main function with output set to True.
    """
    network = VentilationNetwork()
    caculation = Caculation(network)
    main(network, caculation, output=True)
