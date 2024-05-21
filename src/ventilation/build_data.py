from .network import VentilationNetwork
from .caculation import *
import pathlib

def step_data(network: VentilationNetwork, caculation: Caculation, output: bool = False, show=False):
    """Main function to run the calculation.

    This function runs the flow balance calculation, performs concentration calculations,
    and optionally outputs the results to a CSV file.

    Args:
        network (VentilationNetwork): The VentilationNetwork to be used.
        caculation (Caculation): The Caculation object to be used.
        output (bool, optional): Whether to output the results to a CSV file. Defaults to False.

    """
    caculation.flow_balance()
    if show:
        network.visualize_network(show=True)
    result = {f'Node {node.identifier}': {'time': [], 'pressure': [],
                                          'concentration': [], 'people': [], 'size':[]} for node in caculation.nodes}
    for _ in range(3):
        times, concentration_list = caculation.concentration_calculation(
            total_time=200, time_step=5)
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
        outside_node_num = sum(
            [1 for conn in network.connections if conn.node2 == None])
        path = pathlib.Path(__file__).parents[2].joinpath('data')
        path.mkdir(parents=True, exist_ok=True)
        name = f'N{len(network.nodes)}_C{len(network.connections)}_O{outside_node_num}'
        flow = [[conn.node1.identifier,conn.node2.identifier,conn.flow] for conn in network.connections if conn.node2 is not None]
        pd.DataFrame(flow, columns=['Node1', 'Node2', 'Flow']).to_csv(
            path.joinpath(name + '_flow.csv'), index=False)
        caculation.output_result(result).to_csv(
            path.joinpath(name + '.csv'), index=False)
        pd.DataFrame(network.connection_matrix).to_csv(
            path.joinpath(name + '_matrix.csv'), index=False)
        
def multi_step_data(step:int = 10, output: bool = False, show=False):
    for _ in range(step):
        try:
            Node.id_counter = 0
            Connection.id_counter = 0
            network = VentilationNetwork()
            caculation = Caculation(network)
            step_data(network, caculation, output=output, show=show)
        except:
            print('Error')
            continue

if __name__ == '__main__':
    multi_step_data(step=10, output=True)