import numpy as np
from typing import List
from node import Node
from connection import Connection
from network import VentilationNetwork
from scipy.optimize import fsolve, root, differential_evolution


def flow_balance(pressure_list:list,VentilationNetwork: VentilationNetwork) -> list:
    equations = []
    nodes = VentilationNetwork.nodes
    connections = VentilationNetwork.connections
    for node in nodes:
        node.update_pressure(pressure_list[node.identifier])
    for conn in connections:
        conn.update_flow()
    for node in nodes:
        flow = 0
        for conn in connections:
            if conn.node1 == node:
                flow -= conn.flow
            elif conn.node2 == node:
                flow += conn.flow
        equations.append(flow)
    return equations



def main():
    network = VentilationNetwork(1)
    pressure_list = [0] * len(network.nodes)
    result = root(flow_balance, pressure_list, args=(network,))
    if result.success:
        print('解：', result.x)
    else:
        print('找不到解。')
    network.visualize_network()
    pass

if __name__ == '__main__':
    main()