from network import Network
import numpy as np

class Network_create:
    def __init__(self,node_num,outdoor_connection_num,seed=None):
        self.node_num = node_num
        self.ocn = outdoor_connection_num
        if seed:
            np.random.seed(seed)

    def create_network(self):
        connection_num = self.node_num + np.random.randint(1,3)
        network = Network()
        # 添加节点
        for i in range(self.node_num):
            network.add_node(i)
        # 添加室内连接
        for i in range(len(network.nodes) - 1):
            network.add_connection(network.nodes[i],np.random.choice(network.nodes[i+1:]),resistance= round((0.2 + 0.8 * np.random.rand()),2))
            connection_num -= 1
        for i in range(connection_num):
            # 从室内节点中随机选择两个不相同的节点
            node1,node2 = np.random.choice(network.nodes,2,replace=False)
            network.add_connection(node1,node2,resistance=round((0.2 + 0.8 * np.random.rand()),2))
        # 添加室外连接
        for i in range(self.ocn):
            nodes = np.random.choice(network.nodes,self.ocn,replace=False)
            for node in nodes:
                network.add_connection(node,None,resistance=round((0.2 + 0.8 * np.random.rand()),2),external_pressure=np.random.randint(5,15))
        return network
    
    def dev_output(self):
        network = self.create_network()
        network.calculate_network()
        network.display_results()
        return network

if __name__ == "__main__":
    network = Network_create(5,2,1)
    network = network.create_network()
    network.calculate_network()
    network.display_results()