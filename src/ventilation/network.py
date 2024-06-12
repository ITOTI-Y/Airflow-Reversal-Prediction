import numpy as np
from .node import Node
from .connection import Connection
from ..config import CALCULATE_CONFIG
import matplotlib.pyplot as plt
import networkx as nx

ENV_CONFIG = CALCULATE_CONFIG()

class VentilationNetwork:
    """A class used to represent a Ventilation Network.

    Attributes:
        nodes (list): A list of Node objects representing the rooms in the network.
        connections (list): A list of Connection objects representing the connections between rooms.

    Methods:
        _create_random_network(): Creates a random network of nodes and connections.
        _build_connection_matrix(): Builds a connection matrix for the network.
        random_people_number(): Generates a list of random people number for each node.
        draw_multiedge_labels(pos, edge_labels, ax, font_size): Draws labels for multiple edges on a graph.
        visualize_network(): Visualizes the network by creating a graph representation of the nodes and connections.
    """

    def __init__(self, seed: int = None):
        """Initializes a VentilationNetwork with an optional seed for random number generation.

        Args:
            seed (int, optional): The seed for random number generation. Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)
        self.nodes = []  # 房间列表
        self.connections = []  # 连接列表
        self._create_random_network()
        self.connection_matrix = self._build_connection_matrix()

    def _create_random_network(self):
        """Creates a random network of nodes and connections.

        The method generates a random network by creating a specified number of nodes
        and connecting them with random coefficients. It also assigns random outside
        pressures to some connections.
        """
        node_num = np.random.randint(*ENV_CONFIG.NODE_NUM_RANGE)
        connection_num = (np.random.uniform(*ENV_CONFIG.CONNECTION_NUM_TIMES) * node_num) // 1
        outside_pressure = np.random.randint(*ENV_CONFIG.OUTSIDE_PRESSURE_RANGE, np.random.randint(*ENV_CONFIG.OUTSIDE_NODE_NUM))
        for _ in range(node_num):
            self.nodes.append(Node(size=np.random.randint(
                *ENV_CONFIG.NODE_SIZE_RANGE), people=np.random.randint(*ENV_CONFIG.NODE_PEOPLE_RANGE)))

        # chosen_nodes = np.random.choice(self.nodes,np.random.randint(2,5),replace=False)
        for i in range(node_num):
            if i == 0:
                last_node = i
            else:
                coefficient = round(np.random.uniform(*ENV_CONFIG.COEFFICIENT_RANGE), 1)
                self.connections.append(Connection(
                    self.nodes[last_node], self.nodes[i], coefficient))
                last_node = i
                connection_num -= 1
        if connection_num != 0:
            for i in range(int(connection_num)):
                coefficient = round(np.random.uniform(*ENV_CONFIG.COEFFICIENT_RANGE), 1)
                node1, node2 = np.random.choice(self.nodes, 2, replace=False)
                self.connections.append(Connection(node1, node2, coefficient))
        for i in range(outside_pressure.size):
            coefficient = round(np.random.uniform(*ENV_CONFIG.COEFFICIENT_RANGE), 1)
            self.connections.append(Connection(np.random.choice(
                self.nodes), None, coefficient, outside_pressure[i]))

    def _build_connection_matrix(self):
        """Builds a connection matrix for the network.

        The method builds a connection matrix for the network, where each entry represents a connection between two nodes.
        """
        connection_matrix = np.identity(len(self.nodes)+1)  # create a diagonal matrix contaning outdoor nodes
        for conn in self.connections:
            if conn.node2 is not None:
                connection_matrix[conn.node1.identifier][conn.node2.identifier] = 1
                connection_matrix[conn.node2.identifier][conn.node1.identifier] = 1
            else:
                connection_matrix[conn.node1.identifier][-1] = 1
                connection_matrix[-1][conn.node1.identifier] = 1
        return connection_matrix

    def random_people_number(self):
        """Generates a list of random people number for each node.

        The method generates a list of random people number for each node in the network.
        """
        lenght = len(self.nodes)
        sum_people = sum([node.people for node in self.nodes])
        people_list = np.zeros(lenght)
        while np.sum(people_list) < sum_people:
            people_list[np.random.randint(lenght)] += 1

        for i, node in enumerate(self.nodes):
            node.update_people(int(people_list[i]))

    def draw_multiedge_labels(self, pos: dict, edge_labels: dict, ax, font_size: int) -> None:
        """Draws labels for multiple edges on a graph.

        Args:
            pos (dict): A dictionary of node positions.
            edge_labels (dict): A dictionary of edge labels.
            ax (matplotlib.axes.Axes): The axes on which to draw the labels.
            font_size (int): The font size of the labels.
        """
        for (u, v, key, rad), label in edge_labels.items():
            # Calculate the midpoint of the edge for label placement
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            # Calculate the direction of curvature (left or right)
            label_x = x + 0.7 * rad * (y2 - y1)
            label_y = y - 0.7 * rad * (x2 - x1)
            # Draw the label at the midpoint of the edge
            ax.text(label_x, label_y, label, size=font_size,
                    ha='center', va='center')

    def visualize_network(self, show: bool = False, save_path: str = None):
        """Visualizes the network by creating a graph representation of the nodes and connections.

        Args:
            save_path (str, optional): The path to save the visualization. Defaults to None.
        """
        fig, ax = plt.subplots()
        outside_index = 0
        edge_colors = []
        G = nx.MultiDiGraph()
        for _, node in enumerate(self.nodes):
            G.add_node(
                node.identifier, pressure=f'{node.pressure:.2f}', concentration=f'{node.concentration:.0f}', people=f'{node.people}', size=f'{node.size}')
        for _, connection in enumerate(self.connections):
            if connection.node2 is not None:
                G.add_edge(connection.node1.identifier,
                           connection.node2.identifier, flow=f'{connection.flow:.2f}')
            else:
                G.add_node(f'Outside {outside_index}',
                           pressure=connection.outside_pressure, concentration=connection.outside_concentration, people=0, size=0)
                G.add_edge(connection.node1.identifier,
                           f'Outside {outside_index}', flow=f'{connection.flow:.2f}')
                outside_index += 1
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=1000)
        rad = 0.1
        last_edge = None
        edge_labels = {}
        for i, (u, v, key, data) in enumerate(G.edges(keys=True, data=True), start=1):
            if last_edge == (u, v):
                rad += 0.2
            else:
                rad = 0.1
            last_edge = (u, v)
            nx.draw_networkx_edges(G, pos, edgelist=[(
                u, v)], width=3, alpha=0.5, edge_color=f"C{i}", style="solid", connectionstyle=f"arc3,rad={rad}", arrowsize=20)
            edge_labels[(u, v, key, rad)] = data["flow"]
        labels = {
            node: f"Node {node}\n{attrs['pressure']} Pa\n{attrs['concentration']} PPM\n{attrs['size']} m^3" for node, attrs in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        self.draw_multiedge_labels(pos, edge_labels, ax, font_size=8)
        plt.axis('off')
        if save_path:
            plt.savefig(f'{save_path}')
        if show:
            plt.show()


if __name__ == '__main__':
    """Main entry point of the script.

    This block of code is executed if the script is run directly. It creates a VentilationNetwork,
    and visualizes it.
    """
    def main():
        """Main function to run the script.

        This function creates a VentilationNetwork with a seed of 1, and visualizes it.
        """
        network = VentilationNetwork(1)
        network.visualize_network()
    main()
