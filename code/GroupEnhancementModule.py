class SpatialGroupEnhance_for_1D(nn.Module):
    def __init__(self, groups=16):
        super(SpatialGroupEnhance_for_1D, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = x.reshape(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.reshape(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.reshape(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.reshape(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.reshape(b, c, h)
        return x