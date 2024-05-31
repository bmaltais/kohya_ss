def add_spp_layer(spp_layer,sample: torch.FloatTensor) -> torch.FloatTensor:
    sample = spp_layer(sample)
    return spp_layer
def addmaxpool(max_pool_layer,sample: torch.FloatTensor) -> torch.FloatTensor:
    return max_pool_layer(sample)
def make_max_pool() -> torch.FloatTensor:
    max_pool_layer = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    return max_pool_layer
def make_spp_layer(layers):
    spp_layer = nn.Sequential(
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # 3x3 max pooling
        nn.AvgPool2d(kernel_size=5, stride=1, padding=2),  # 5x5 average pooling
        nn.AvgPool2d(kernel_size=7, stride=1, padding=3),  # 7x7 max pooling
        nn.AvgPool2d(kernel_size=9, stride=1, padding=4),  # 9x9 average pooling            #nn.Conv2d(sample.shape[1], sample.shape[1] // 2, kernel_size=1),  # 1x1 convolution
    )
    return spp_layer
    
