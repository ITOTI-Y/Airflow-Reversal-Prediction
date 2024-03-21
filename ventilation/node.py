class Node:
    def __init__(self,identifier, size, initial_pressure=0, initial_co2=400):
        self.identifier = identifier
        self.size = size  # 房间尺寸
        self.pressure = initial_pressure  # 初始风压
        self.co2 = initial_co2  # 初始二氧化碳浓度

    def update_pressure(self, new_pressure):
        # 更新房间风压
        self.pressure = new_pressure

    def __repr__(self):
        respond = f'Node {self.identifier}, size: {self.size}, pressure: {self.pressure}, co2: {self.co2}'
        return respond