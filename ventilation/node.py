class Node:
    id_counter = 0

    def __init__(self, size: float, people: int, initial_pressure: float = 0.0, initial_concentration: float = 400.0):
        self.identifier = Node.id_counter
        Node.id_counter += 1
        self.size = size  # 房间尺寸 m^3
        self.people = people  # 人数 人
        self.pressure = initial_pressure  # 初始风压 Pa
        self.concentration = initial_concentration  # 初始浓度 ppm

    def update_pressure(self, new_pressure:float):
        """
        Updates the air pressure in the room.

        Args:
            new_pressure (float): The new air pressure value.

        """
        self.pressure = new_pressure

    def update_people(self, new_people:int):
        """
        Updates the people in the room.

        Args:
            new_people (int): The new people value.

        """
        self.people = new_people

    def update_concentration(self, new_concentration:float):
        """
        Updates the concentration in the room.

        Args:
            new_concentration (float): The new concentration value.

        """
        self.concentration = new_concentration

    def __repr__(self):
        respond = f'Node {self.identifier}, size: {self.size}, pressure: {self.pressure}, concentration: {self.concentration}'
        return respond
