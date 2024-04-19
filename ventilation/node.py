class Node:
    id_counter = 0

    def __init__(self, size: float, people: int, initial_pressure: float = 0.0, initial_concentration: float = 400.0):
        self.identifier = Node.id_counter
        Node.id_counter += 1
        self.size = size  # 房间尺寸 m^3
        self.people = people  # 人数 人
        self.pressure = initial_pressure  # 初始风压 Pa
        self.concentration = initial_concentration  # 初始浓度 ppm

    def update_pressure(self, new_pressure: float):
        """
        更新房间的空气压力。

        Args:
            new_pressure (float): 新的空气压力值。

        """
        self.pressure = new_pressure

    def update_people(self, new_people: int):
        """
        更新房间的人数。

        Args:
            new_people (int): 新的人数值。

        """
        self.people = new_people

    def update_concentration(self, new_concentration: float):
        """
        更新房间的浓度。

        Args:
            new_concentration (float): 新的浓度值。

        """
        self.concentration = new_concentration

    def __repr__(self) -> str:
        return f'Node {self.identifier}, size: {self.size}, pressure: {self.pressure}, concentration: {self.concentration}'