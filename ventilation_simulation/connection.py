from node import Node
import numpy as np

class Connection:
    def __init__(self,node1,node2,resistance,external_pressure=None):
        """
        连接初始化
        :param node1: 第一个节点对象（假设为室内节点）
        :param node2: 第二个节点对象，可以是室外节点，此参数可选
        :param resistance: 连接的阻力
        :param external_pressure: 如果第二个节点是室外节点，需要提供室外压力
        """
        self.node1 = node1
        self.node2 = node2 # node2可能为None,表示这是一个到室外的连接
        self.resistance = resistance
        self.external_pressure = external_pressure if node2 is None else None # 仅当连接到室外时使用
        self.flow = 0.0

    def calculate_flow(self):
        """
        根据当前节点的气压差和外界气压计算流量
        """
        if self.node2: # 如果存在第二个节点，意味着室内到室内的连接
            p1 = self.node1.pressure
            p2 = self.node2.pressure
        else: # 室内到室外的连接
            p1 = self.node1.pressure
            p2 = self.external_pressure

        delta_p = p1 - p2
        self.flow = self.resistance * np.sqrt(abs(delta_p)) * np.sign(delta_p)

    def __repr__(self):
        return f"Connection({self.node1.identifier} <-> {'External' if self.node2 is None else self.node2.identifier}, Flow: {self.flow})"
    

if __name__ == "__main__":
    # 创建室内节点
    node1 = Node("Room1")
    node2 = Node("Room2")

    # 创建与室外的连接
    connection1 = Connection(node1, None, 0.5, external_pressure=10)  # 室外气压为10pa
    connection2 = Connection(node1, None, 0.5, external_pressure=8)  # 室外气压为8pa
    connection3 = Connection(node1,node2,0.5)  # 室内到室内的连接

    # 计算流量（在实际应用中，您需要在迭代过程中更新节点的气压值）
    connection1.calculate_flow()
    connection2.calculate_flow()
    connection3.calculate_flow()

    # 打印连接信息
    print(f"Connection 1: {connection1}")
    print(f"Connection 2: {connection2}")
    print(f"Connection 3: {connection3}")