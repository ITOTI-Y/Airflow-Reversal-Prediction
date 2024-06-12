import numpy as np
from .node import Node
from ..config import CALCULATE_CONFIG

ENV_CONFIG = CALCULATE_CONFIG()

class Connection:
    """A class used to represent a Connection in a Ventilation Network.

    Attributes:
        id_counter (int): A counter to assign unique identifiers to each connection.
        identifier (int): The unique identifier of the connection.
        node1 (Node): The first node (room) of the connection.
        node2 (Node): The second node (room) of the connection. None if the connection is with the outside.
        coefficient (float): The resistance coefficient of the connection.
        outside_pressure (float): The outside air pressure in Pa. None if the connection is not with the outside.
        outside_concentration (float): The outside CO2 concentration in ppm. Defaults to 400.
        flow (float): The air flow in the connection. Positive values indicate flow from node1 to node2.

    Methods:
        calculate_flow(pressure1: float, pressure2: float) -> float: Calculates the air flow based on the pressure difference.
        update_flow(): Updates the air flow in the connection.
    """
    id_counter = 0

    def __init__(self, node1: Node, node2: Node, coefficient: float, outside_pressure: float = None):
        """Initializes a Connection with the given nodes, coefficient, and optional outside pressure.

        Args:
            node1 (Node): The first node (room) of the connection.
            node2 (Node): The second node (room) of the connection. None if the connection is with the outside.
            coefficient (float): The resistance coefficient of the connection.
            outdoor_pressure (float, optional): The outside air pressure in Pa. None if the connection is not with the outside.
        """
        self.identifier = Connection.id_counter
        Connection.id_counter += 1
        self.node1 = node1  # 连接的房间1
        self.node2 = node2  # 连接的房间2
        self.coefficient = coefficient  # 阻力系数
        self.outside_pressure = outside_pressure  # 室外风压 Pa
        self.outside_concentration = ENV_CONFIG.OUTSIDE_CONCENTRATION  # 室外浓度 ppm
        self.flow = 0  # 气流流量,正值表示从room1流向room2,负值表示从room2流向room1

    def calculate_flow(self, pressure1: float, pressure2: float) -> float:
        """Calculates the air flow based on the pressure difference.

        The air flow is calculated using the formula Q=C*sqrt(ΔP), where Q is the air flow, C is the resistance coefficient,
        and ΔP is the pressure difference.

        Args:
            pressure1 (float): The air pressure in the first node (room).
            pressure2 (float): The air pressure in the second node (room) or the outside pressure if the connection is with the outside.

        Returns:
            float: The calculated air flow.
        """
        pressure_diff = pressure1 - pressure2
        result = self.coefficient * \
            np.sign(pressure_diff) * np.sqrt(abs(pressure_diff))
        return result

    def update_flow(self):
        """Updates the air flow in the connection.

        The method updates the air flow in the connection by calculating the flow based on the current pressures in the nodes.
        """
        if self.node2 is None:
            self.flow = self.calculate_flow(
                self.node1.pressure, self.outside_pressure)
        else:
            self.flow = self.calculate_flow(
                self.node1.pressure, self.node2.pressure)
    
    def reset_id_counter():
        """Resets the id_counter to 0."""
        Connection.id_counter = 0

    def __repr__(self) -> str:
        """Returns a string representation of the Connection.

        Returns:
            str: A string representation of the Connection.
        """
        return f"ID {self.identifier} Node {self.node1.identifier} <-> {'Node' + str(self.node2.identifier) if self.node2 else 'None'}, coefficient: {self.coefficient}, flow: {self.flow}"
