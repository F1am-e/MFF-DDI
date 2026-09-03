def skip_node_mask(N, skip_rate=0.5, skip_type="uniform", degree=None, device="cpu"):

    mask = torch.FloatTensor([1.0 for _ in range(N)])

    if skip_type == 'degree':

        if degree is None:
            print("[SkipNode Warning] 'degree' not provided, falling back to 'uniform' skip_type.")
            prob = torch.ones(N) / N
        else:
            prob = degree / (degree.sum() + 1e-8)
    else:

        prob = torch.ones(N) / N

    index = torch.Tensor([i for i in range(N)]).to(device)
    size = int(N * skip_rate)


    if size <= 0 or size >= N:

        if skip_rate == 0.0:
            return mask.unsqueeze(1).to(device)
        else:

            mask.fill_(0.0)
            return mask.unsqueeze(1).to(device)

    dataloader = DataLoader(dataset=index, batch_size=size,
                            sampler=WeightedRandomSampler(prob, size, replacement=False))
    sampled_idx = None
    for data in dataloader:
        sampled_idx = data
    sampled_idx = sampled_idx.to(torch.int64).cpu()
    mask = mask.index_fill_(0, sampled_idx, 0)
    mask = mask.unsqueeze(1).to(device)
    return mask


def skip_node(x_old, x_new, skip_rate=0.5, skip_type="uniform", degree=None, device="cpu"):

    if skip_rate == 0.0:
        return x_new

    mask = skip_node_mask(x_old.shape[0], skip_rate=skip_rate, skip_type=skip_type, degree=degree, device=device)
    out = mask * (x_new - x_old) + x_old

    return out