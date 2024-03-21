import numpy as np

class Connection:
    def __init__(self, node1, node2, coefficient, outdoor_pressure=0):
        
        self.node1 = node1  # 连接的房间1
        self.node2 = node2  # 连接的房间2
        self.coefficient = coefficient  # 阻力系数
        self.outside_pressure = outdoor_pressure  # 室外风压
        self.flow = 0  # 气流流量,正值表示从room1流向room2,负值表示从room2流向room1

    def calculate_flow(self, pressure1, pressure2):
        # 根据风压差计算气流流量,使用Q=C*sqrt(ΔP)
        pressure_diff = pressure1 - pressure2
        result = self.coefficient * np.sign(pressure_diff) * np.sqrt(abs(pressure_diff))
        return self.coefficient * np.sign(pressure_diff) * np.sqrt(abs(pressure_diff))

    def update_flow(self):
        # 更新气流流量
        if self.node2 is None:  # 与室外相连
            self.flow = self.calculate_flow(
                self.node1.pressure, self.outside_pressure)
        else:  # 与另一个房间相连
            self.flow = self.calculate_flow(
                self.node1.pressure, self.node2.pressure)

    def __repr__(self):
        respond = f"Node {self.node1.identifier} <-> {'Node' + str(self.node2.identifier) if self.node2 else 'None'}, coefficient: {self.coefficient}, flow: {self.flow}"
        return respond
