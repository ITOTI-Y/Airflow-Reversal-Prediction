import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from node import Node
from connection import Connection
import matplotlib.pyplot as plt
import networkx as nx

class VentilationNetwork:
    def __init__(self,seed=None):
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.nodes = []  # 房间列表
        self.connections = []  # 连接列表
        self.create_random_network()

    def create_random_network(self):
        node_num = np.random.randint(4,10)
        connection_num = (np.random.uniform(1.5,2) * node_num) // 1
        outside_pressure = np.random.randint(1,10,3)
        for i in range(node_num):
            self.nodes.append(Node(i,np.random.randint(45,60)))
        for i in range(node_num):
            if i == 0:
                last_node = i
            else:
                coefficient = round(np.random.uniform(0.2,0.8),1)
                self.connections.append(Connection(self.nodes[last_node],self.nodes[i],coefficient))
                last_node = i
                connection_num -= 1
        if connection_num != 0:
            for i in range(int(connection_num)):
                coefficient = round(np.random.uniform(0.2,0.8),1)
                node1,node2 = np.random.choice(self.nodes,2,replace=False)
                self.connections.append(Connection(node1,node2,coefficient))
        for i in range(outside_pressure.size):
            coefficient = round(np.random.uniform(0.2,0.8),1)
            self.connections.append(Connection(np.random.choice(self.nodes),None,coefficient,outside_pressure[i]))

    def draw_multiedge_labels(self,G, pos, edge_labels, ax, font_size):
        for (u, v, key, rad), label in edge_labels.items():
            # 计算边的中点位置，用于放置标签
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            # 计算曲率的方向（左侧或右侧）
            label_x = x + 0.7 * rad * (y2 - y1)
            label_y = y - 0.7 * rad * (x2 - x1)
            # 在边的中点位置上绘制标签
            ax.text(label_x, label_y, label, size=font_size, ha='center', va='center')

    def visualize_network(self):
        fig, ax = plt.subplots()
        outside_index = 0
        edge_colors = []
        G = nx.MultiDiGraph()
        for _,node in enumerate(self.nodes):
            G.add_node(node.identifier,pressure = f'{node.pressure:.2f}',co2 = f'{node.co2:.2f}')
        for _,connection in enumerate(self.connections):
            if connection.node2 is not None:
                G.add_edge(connection.node1.identifier,connection.node2.identifier,flow = f'{connection.flow:.2f}')
            else:
                G.add_node(f'Outside {outside_index}',pressure = connection.outside_pressure)
                G.add_edge(connection.node1.identifier,f'Outside {outside_index}',flow = f'{connection.flow:.2f}')
                outside_index += 1
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=1000)
        rad = 0.1
        last_edge = None
        edge_labels = {}
        for i, (u, v, key, data) in enumerate(G.edges(keys=True,data=True), start=1):
            if last_edge == (u,v):
                rad += 0.2
            else:
                rad = 0.1
            last_edge = (u,v)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3, alpha=0.5, edge_color=f"C{i}", style="solid",connectionstyle=f"arc3,rad={rad}",arrowsize=20)
            edge_labels[(u,v,key,rad)] = data["flow"]
        labels = {node: f"{node}\n{attrs['pressure']} Pa" for node, attrs in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        self.draw_multiedge_labels(G, pos, edge_labels, ax, font_size=8)
        plt.axis('off')
        plt.show()

def main():
    network = VentilationNetwork(1)
    network.visualize_network()
    pass

if __name__ == '__main__':
    main()