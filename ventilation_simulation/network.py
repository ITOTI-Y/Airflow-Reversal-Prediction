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
    
    @staticmethod
    def io_flow(connections,node):
        """
        输出特定节点进出口的流量
        :param connection: 连接
        :param node: 节点
        :return: 进口流量，出口流量
        """
        in_flow = 0
        out_flow = 0
        for c in connections:
            if c.node1 == node:
                if c.flow < 0:
                    in_flow += abs(c.flow)
                else:
                    out_flow += abs(c.flow)
            elif c.node2 == node:
                if c.flow < 0:
                    out_flow += abs(c.flow)
                else:
                    in_flow += abs(c.flow)
        return in_flow,out_flow
    
    def pressure_update_method(self,node,airflow_difference):
        """
        气压更新模型
        :param node: 节点
        :param airflow_difference: 流量差
        """
        new_pressure = node.pressure + airflow_difference * 0.1
        pressure_difference = new_pressure - node.pressure
        node.update_pressure(new_pressure)
        return new_pressure,pressure_difference

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
                in_flow,out_flow = self.io_flow(self.connections,node)
                net_flow = in_flow - out_flow

                # 气压更新模型
                new_pressure,pressure_difference = self.pressure_update_method(node,net_flow)
                pressure_changes.append(abs(pressure_difference))

            # 检查是否到达收敛条件
            if max(pressure_changes,default=0) < tolerance:
                print(f"Converged after {i} iterations")
                break
    
    def display(self):
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

    def build_random_network(self,node_num,outdoor_connection_num,seed=None):
        """
        创建一个随机网络
        :param node_num: 节点数量
        :param outdoor_connection_num: 室外连接数量
        :param seed: 随机数种子
        """
        if seed:
            np.random.seed(seed)

        connection_num = node_num + np.random.randint(1,3)
        # 添加节点
        for i in range(node_num):
            self.add_node(f"Room{i+1}")
        # 添加室内连接
        for i in range(len(self.nodes) - 1):
            self.add_connection(self.nodes[i],np.random.choice(self.nodes[i+1:]),resistance= round((0.2 + 0.8 * np.random.rand()),2))
            connection_num -= 1
        for i in range(connection_num):
            # 从室内节点中随机选择两个不相同的节点
            node1,node2 = np.random.choice(self.nodes,2,replace=False)
            self.add_connection(node1,node2,resistance=round((0.2 + 0.8 * np.random.rand()),2))
        # 添加室外连接
        nodes = np.random.choice(self.nodes,outdoor_connection_num,replace=False)
        for node in nodes:
            self.add_connection(node,None,resistance=round((0.2 + 0.8 * np.random.rand()),2),external_pressure=np.random.randint(5,15))

        return self


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
    network.calculate_network(iterations=2000,tolerance=1e-6)
    network.display()