import pytest
from ventilation.node import Node

class TestNode:
    @pytest.fixture
    def node(self):
        return Node(size=100, people=5, initial_pressure=1000, initial_concentration=500)

    def test_update_pressure(self, node):
        node.update_pressure(2000)
        assert node.pressure == 2000

    def test_update_people(self, node):
        node.update_people(10)
        assert node.people == 10

    def test_update_concentration(self, node):
        node.update_concentration(600)
        assert node.concentration == 600

    def test_reset_id_counter(self):
        Node.reset_id_counter()
        assert Node.id_counter == 0

    def test_repr(self, node):
        expected_repr = "Node 0, size: 100, pressure: 1000, concentration: 500"
        assert repr(node) == expected_repr