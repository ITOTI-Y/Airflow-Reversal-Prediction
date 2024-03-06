import numpy as np

class Node:
    def __init__(self,identifier):
        """
        初始化节点
        :param identifier: 节点ID
        :param external_pressure: 室外压力
        """
        self.identifier = identifier
        self.pressure = 0.0

    def update_pressure(self,new_pressure):
        """
        更新压力
        :param new_pressure: 新压力
        """
        self.pressure = new_pressure

    def __repr__(self): # 当print的使用使用该函数用于输出
        return f"Node({self.identifier}, pressure={self.pressure})"
    
if __name__ == "__main__":
    # 创建示例节点
    node1 = Node("Room1")
    node2 = Node("Room2")
    print(f"Node 1: {node1}")
    print(f"Node 2: {node2}")