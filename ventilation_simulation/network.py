from node import Node
from connection import Connection
import numpy as np

class Network:
    def __init__(self):
        """
        初始化网络
        """
        self.nodes = []
        self.connections = []

    def add_node(self,identifier):
        """
        添加一个新节点到网络
        :param identifier: 节点ID
        :param is_external: 是否与室外相连
        :param external_pressure: 室外压力
        """

        node = Node(identifier)
        self.nodes.append(node)
        return node
    
    def add_connection(self,node1,node2,resistance,external_pressure=None):
        """
        添加一个新连接到网络
        :param node1: 第一个节点对象
        :param node2: 第二个节点对象
        :param resistance: 连接的阻力
        """
        connection = Connection(node1,node2,resistance,external_pressure)
        self.connections.append(connection)
        return connection
    
    def calculate_network(self,iterations=5000,tolerance=1e-6):
        """
        通过迭代计算网络中气压和流量
        :param iterations: 迭代次数
        :param tolerance: 容差
        """
        for i in range(iterations):
            # 计算每个连接的流量
            for connection in self.connections:
                connection.calculate_flow()

            pressure_changes = []
            for node in self.nodes:
                # 对于每个室内节点，基于流入和流出的流量计算气压变化
                inflow = 0
                outflow = 0
                for c in self.connections:
                    if c.node1 == node:
                        if c.flow < 0:
                            inflow += abs(c.flow)
                        else:
                            outflow += abs(c.flow)
                    elif c.node2 == node:
                        if c.flow < 0:
                            outflow += abs(c.flow)
                        else:
                            inflow += abs(c.flow)
                net_flow = inflow - outflow

                # 气压更新模型
                new_pressure = node.pressure + net_flow * 0.1
                pressure_changes.append(abs(new_pressure - node.pressure))
                node.update_pressure(new_pressure)

            # 检查是否到达收敛条件
            if max(pressure_changes,default=0) < tolerance:
                print(f"Converged after {i} iterations")
                break

    
    def display_results(self):
        """
        打印网络中的节点和连接信息
        """
        print("节点气压:")
        for node in self.nodes:
            print(f"{node}")
        print("\n连接流量:")
        for connection in self.connections:
            print(f"{connection}")
        print("\n节点流量差:")
        for node in self.nodes:
            flow_rate_error = 0
            for c in self.connections:
                if c.node1 == node:
                    flow_rate_error += c.flow
                elif c.node2 == node:
                    flow_rate_error -= c.flow
            print(f"{node.identifier}: {flow_rate_error}")

if __name__ == "__main__":
    network = Network()
    # 示例：创建室内节点和室外连接
    room1 = network.add_node("Room1")
    room2 = network.add_node("Room2")
    room3 = network.add_node("Room3")
    external_pressure = 10  # 假定的室外气压

    # 创建室内到室外的连接
    network.add_connection(room1, None, 0.5, external_pressure)
    network.add_connection(room2, None, 0.5, external_pressure - 5)  # 不同的外部气压示例

    # 创建室内节点之间的连接
    network.add_connection(room1, room2, 0.3)
    network.add_connection(room1, room2, 0.4)
    network.add_connection(room1, room3, 0.4)
    network.add_connection(room2, room3, 0.2)

    # 运行计算
    network.calculate_network(iterations=1000)
    network.display_results()