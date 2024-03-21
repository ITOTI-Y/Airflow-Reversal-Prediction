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
        self.concentration = 440.0 # 浓度 ppm
        self.volume = 45
        self.people = 0

    def update_pressure(self,new_pressure):
        """
        更新压力
        :param new_pressure: 新压力
        """
        self.pressure = new_pressure
    
    def update_concentration(self,new_concentration):
        """
        更新CO2 ppm
        :param new_ppm: CO2 ppm
        """
        self.concentration = new_concentration

    def update_volume(self,new_volume):
        """
        更新体积
        :param new_volume: 新体积
        """
        self.volume = new_volume

    def update_people(self,new_people):
        """
        更新人数
        :param new_people: 新人数
        """
        self.people = new_people

    def __repr__(self): # 当print时使用该函数用于输出
        return f"Node({self.identifier}, pressure={self.pressure}, CO2={self.concentration}ppm, People={self.people}, Volume={self.volume}m^3)"
    
if __name__ == "__main__":
    # 创建示例节点
    node1 = Node("Room1")
    node2 = Node("Room2")
    print(f"Node 1: {node1}")
    print(f"Node 2: {node2}")