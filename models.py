class Room:
    """表示一个房间，可包含多个门窗"""
    def __init__(self,id):
        self.id = id
        self.connections = []

class Connection:
    """表示房间之间的连接,包括门窗"""
    def __init__(self,from_room,to_room,connection_type):
        self.from_room = from_room
        self.to_room = to_room
        self.type = connection_type
        self.resistance = 0.8 if self.type == 'door' else 0.5

class VentilationNetwork:
    """表示整个通风网络"""
    def __init__(self):
        self.rooms = {}
        self.connections = []

    def add_room(self,room_id):
        """添加房间到网络"""
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(room_id)

    def add_connection(self,from_room_id,to_room_id,connection_type):
        """添加连接到网络"""
        self.add_room(from_room_id)
        self.add_room(to_room_id)
        connection = Connection(from_room_id,to_room_id,connection_type)
        self.connections.append(connection)
        self.rooms[from_room_id].connections.append(connection)
        if to_room_id: # to_room_id 为None表示室外
            self.rooms[to_room_id].connections.append(connection)