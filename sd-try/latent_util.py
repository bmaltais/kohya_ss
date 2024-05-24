import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 accelerator.device 是一个已经定义的设备变量
# 如果没有定义，可以将其替换为 'cuda' 如果在GPU上运行，或者 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_device(device):
    self.device = device

def apply_convolution(input_data, device):
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device).float()
    return conv(input_data)

def apply_pooling(input_data, device):
    # 注意：这里假设 nn.LPool2d 是一个有效的层，但实际上 torch.nn 没有这个层
    # 如果需要一个 L2 池化层，可能需要自定义一个层或者使用其他方法
    lp_pool = nn.LPPool2d(2, kernel_size=2, stride=2, ceil_mode=False).to(device).float()
    return lp_pool(input_data)

def apply_upsampling(input_data, device):
    return F.interpolate(input_data, scale_factor=2, mode='bilinear', align_corners=True).to(device).float()

def make_latents(noisy_latents,dim,device):  
        # 分割张量
    split_size = noisy_latents.size(dim) // 2
    part1, part2 = torch.split(noisy_latents, split_size, dim=dim)
    """
    print("noisy_latents:", noisy_latents.shape)
    print("part1:", part1.shape)
    print("part2:", part2.shape)
    """
    # 卷积操作
    conv_output1 = apply_convolution(part1, device)
    conv_output2 = apply_convolution(part2, device)
    """
    print("conv_output1:", conv_output1.shape)
    print("conv_output2:", conv_output2.shape)
    """
    # 池化操作
    l2_pooled_output1 = apply_pooling(conv_output1, device)
    l2_pooled_output2 = apply_pooling(conv_output2, device)
    """
    print("l2_pooled_output1:", l2_pooled_output1.shape)
    print("l2_pooled_output2:", l2_pooled_output2.shape)
    """
    # 上采样操作
    upsampled_output1 = apply_upsampling(l2_pooled_output1, device)
    upsampled_output2 = apply_upsampling(l2_pooled_output2, device)
    """
    # 验证维度恢复情况
    print("Upsampled Output 1 shape:", upsampled_output1.shape)
    print("Upsampled Output 2 shape:", upsampled_output2.shape)
    """
    # 拼接两个部分
    noisy_latents = torch.cat([upsampled_output1, upsampled_output2], dim=dim)
    
    # 打印最终结果形状
    #print("noisy latents shape:", noisy_latents.shape)
    #print("noisy latents max:", torch.max(noisy_latents))
    return noisy_latents 
def process_noisy_latents(noisy_latent,device,is_for_height = False):
    print("noisy latents shape1:", noisy_latent.shape)
    noisy_latents = make_latents(noisy_latent,-1,device)
    if is_for_height:
        noisy_latents2 = make_latents(noisy_latent,-2,device)
        noisy_latents = torch.cat([noisy_latents, noisy_latents2], dim=-3)
        print("noisy latents shape2:", noisy_latents.shape)
        conv_layer = nn.Conv2d(16, 4, kernel_size=2, stride=2, padding=0).to(device).float()
        noisy_latents = conv_layer(noisy_latents)
    else:
        print("noisy latents shape3:", noisy_latents.shape)
        conv_layer = nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=0).to(device).float()
        noisy_latents = conv_layer(noisy_latents)
    print("noisy latents shape4:", noisy_latents.shape)
    return noisy_latents
