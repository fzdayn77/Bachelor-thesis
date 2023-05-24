import torch

def nt_xnet_loss(z_1, z_2, temperature):
    """
    This is an implementation of the loss function used in the SimCLR paper
    (ArXiv, https://arxiv.org/abs/2002.05709).
    """

    # Concatenates the given sequence of tensors in the given dimension.
    # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    output = torch.cat([z_1, z_2], dim=0)
    num_samples = len(output)

    # Full similarity matrix
    sim = torch.exp(torch.mm(output, output.t().contiguous()) / temperature)

    # Negative similarity
    mask = ~torch.eye(num_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(num_samples, -1).sum(dim=-1)

    # Positive similarity
    pos = torch.exp(torch.sum(z_1 * z_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # Claculating the loss
    loss = -torch.log(pos / neg).mean()

    return loss