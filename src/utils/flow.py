import torch

def flows_to_matrix(flows: torch.Tensor) -> torch.Tensor:
    """
    Compute the net flow of the flows.

    Args:
        flows (torch.Tensor): the flows to compute the net flow

    Returns:
        torch.Tensor: output's device is the same as the input's device
    """
    if not isinstance(flows, torch.Tensor):
        flows = torch.tensor(flows)

    size = int(flows[:, :-1].max().item()) + 1
    flow_matrix = torch.zeros((size,size)).to(flows.device)
    mask = torch.triu(torch.ones(size,size),diagonal=1).to(flows.device)
    for flow in flows:
        flow_matrix[int(flow[0]),int(flow[1])] += flow[2]
    flow_matrix = (flow_matrix - flow_matrix.T)
    flow_matrix = flow_matrix * mask
    return flow_matrix