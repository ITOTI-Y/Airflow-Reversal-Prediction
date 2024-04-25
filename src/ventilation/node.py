class Node:
    """A class used to represent a Node in a Ventilation Network.

    Attributes:
        id_counter (int): A counter to assign unique identifiers to each node.
        identifier (int): The unique identifier of the node.
        size (float): The size of the room in m^3.
        people (int): The number of people in the room.
        pressure (float): The air pressure in the room in Pa.
        concentration (float): The CO2 concentration in the room in ppm.

    Methods:
        update_pressure(new_pressure: float): Updates the air pressure in the room.
        update_people(new_people: int): Updates the number of people in the room.
        update_concentration(new_concentration: float): Updates the CO2 concentration in the room.
        reset_id_counter(): Resets the id_counter to 0.
    """
    id_counter = 0

    def __init__(self, size: float, people: int, initial_pressure: float = 0.0, initial_concentration: float = 400.0):
        """Initializes a Node with the given size, number of people, and optional initial pressure and concentration.

        Args:
            size (float): The size of the room in m^3.
            people (int): The number of people in the room.
            initial_pressure (float, optional): The initial air pressure in the room in Pa. Defaults to 0.0.
            initial_concentration (float, optional): The initial CO2 concentration in the room in ppm. Defaults to 400.0.
        """
        self.identifier = Node.id_counter
        Node.id_counter += 1
        self.size = size
        self.people = people
        self.pressure = initial_pressure
        self.concentration = initial_concentration

    def update_pressure(self, new_pressure: float):
        """Updates the air pressure in the room.

        Args:
            new_pressure (float): The new air pressure in the room in Pa.
        """
        self.pressure = new_pressure

    def update_people(self, new_people: int):
        """Updates the number of people in the room.

        Args:
            new_people (int): The new number of people in the room.
        """
        self.people = new_people

    def update_concentration(self, new_concentration: float):
        """Updates the CO2 concentration in the room.

        Args:
            new_concentration (float): The new CO2 concentration in the room in ppm.
        """
        self.concentration = new_concentration

    def reset_id_counter():
        """Resets the id_counter to 0."""
        Node.id_counter = 0

    def __repr__(self) -> str:
        """Returns a string representation of the Node.

        Returns:
            str: A string representation of the Node.
        """
        return f'Node {self.identifier}, size: {self.size}, pressure: {self.pressure}, concentration: {self.concentration}'
