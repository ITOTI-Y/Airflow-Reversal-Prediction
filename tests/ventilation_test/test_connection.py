import pytest
import numpy as np
from ventilation.connection import Connection
from ventilation.node import Node

class TestConnection:
    @pytest.fixture
    def connection(self):
        Node.reset_id_counter()
        Connection.reset_id_counter()
        node1 = Node(size=100, people=5, initial_pressure=10, initial_concentration=500)
        node2 = Node(size=200, people=10, initial_pressure=20, initial_concentration=1000)
        return Connection(node1=node1, node2=node2, coefficient=0.5, outside_pressure=30)

    def test_calculate_flow(self, connection):
        pressure1 = 10
        pressure2 = 20
        coefficient = 0.5
        expected_flow = coefficient * np.sign(pressure1 - pressure2) * np.sqrt(abs(pressure1 - pressure2))
        assert connection.calculate_flow(pressure1, pressure2) == expected_flow

    def test_update_flow_with_node2(self, connection):
        connection.update_flow()
        expected_flow = 0.5 * np.sign(10 - 20) * np.sqrt(abs(10 - 20))
        assert connection.flow == expected_flow

    def test_update_flow_without_node2(self, connection):
        connection.node2 = None
        connection.update_flow()
        expected_flow = 0.5 * np.sign(10 - 30) * np.sqrt(abs(10 - 30))
        assert connection.flow == expected_flow

    def test_reset_id_counter(self):
        Connection.reset_id_counter()
        assert Connection.id_counter == 0

    def test_repr_with_node2(self, connection):
        connection.update_flow()
        expected_flow = 0.5 * np.sign(10 - 20) * np.sqrt(abs(10 - 20))
        expected_repr = f"ID 0 Node 0 <-> Node1, coefficient: 0.5, flow: {expected_flow}"
        assert repr(connection) == expected_repr

    def test_repr_without_node2(self, connection):
        connection.node2 = None
        connection.update_flow()
        expected_flow = 0.5 * np.sign(10 - 30) * np.sqrt(abs(10 - 30))
        expected_repr = f"ID 0 Node 0 <-> None, coefficient: 0.5, flow: {expected_flow}"
        assert repr(connection) == expected_repr