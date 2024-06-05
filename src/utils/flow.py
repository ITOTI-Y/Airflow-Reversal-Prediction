import torch

def compute_net_flow(flows: torch.Tensor) -> torch.Tensor:
    """
    Compute the net flow of the flows.

    Args:
        flows (torch.Tensor): the flows to compute the net flow

    Returns:
        torch.Tensor: output's device is the same as the input's device
    """
    flow_dict = {}
    if not isinstance(flows, torch.Tensor):
        flows = torch.tensor(flows)
    for flow in flows:
        node1, node2, value = flow[0].item(), flow[1].item(), flow[2].item()
        if node1 == node2:
            continue
        elif (node1,node2) in flow_dict:
            flow_dict[(node1,node2)] += value
        elif (node2,node1) in flow_dict:
            flow_dict[(node2,node1)] -= value
        else:
            flow_dict[(node1,node2)] = value
    result = []
    for i, (node1,node2) in enumerate(flow_dict.keys()):
        result.append(torch.tensor((node1, node2, flow_dict[(node1,node2)])))
    return torch.stack(result).to(flows.device)