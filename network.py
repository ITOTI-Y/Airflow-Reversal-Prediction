from models import VentilationNetwork
import numpy as np

class NetworkBuilder:
    """辅助类，用于构建和管理通风网络"""


    @staticmethod
    def build_network(rooms,connections):
        """构建通风网络"""
        network = VentilationNetwork()
        for room_id in rooms:
            network.add_room(room_id)
        for connection in connections:
            network.add_connection(*connection)
        return network
    

def main():
    rooms = ['Room1','Room2','Room3']
    connections = [('Room1','Room2','door'),('Room2','Room3','window'),('Room3',None,'door')]
    network = NetworkBuilder.build_network(rooms,connections)
    print(network.rooms['Room1'].connections[0].to_room)

if __name__ == '__main__':
    main()