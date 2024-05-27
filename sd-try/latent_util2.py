import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyModel(nn.Module):
    def __init__(self,device):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)
        self.lp_pool = nn.LPPool2d(2, kernel_size=2, stride=2, ceil_mode=False).to(device)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.lp_pool(x)
        return x

class Upsampling(nn.Module):
    def __init__(self):
        super(Upsampling, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

class ProcessLatents(nn.Module):
    def __init__(self,device):
        super(ProcessLatents, self).__init__()
        self.conv1 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1).to(device)
        self.lp_pool = nn.LPPool2d(2, kernel_size=2, stride=2, ceil_mode=False).to(device)
        self.upsampling = Upsampling()

    def forward(self, x, device, is_for_height=False):
        noisy_latents = self.make_latents(x, -1, device)
        if is_for_height:
            noisy_latents2 = self.make_latents(x, -2, device)
            noisy_latents = torch.cat([noisy_latents, noisy_latents2], dim=-3)
            noisy_latents = self.conv1(noisy_latents)
        else:
            noisy_latents = self.conv2(noisy_latents)
        return noisy_latents

    def make_latents(self, noisy_latents, dim, device):
        split_size = noisy_latents.size(dim) // 2
        part1, part2 = torch.split(noisy_latents, split_size, dim=dim)
        print(f"test_noisy_latent:part2:{part2.shape}")
        conv_output1 = self.conv1(part1.to(device))
        conv_output2 = self.conv1(part2.to(device))
        print(f"test_noisy_latent:conv_output2:{conv_output2.shape}")
        l2_pooled_output1 = self.lp_pool(conv_output1)
        l2_pooled_output2 = self.lp_pool(conv_output2)
        print(f"test_noisy_latent:l2_pooled_output2:{l2_pooled_output2.shape}")
        upsampled_output1 = self.upsampling(l2_pooled_output1)
        upsampled_output2 = self.upsampling(l2_pooled_output2)
        print(f"test_noisy_latent:upsampled_output2:{upsampled_output2.shape}")
        noisy_latents = torch.cat([upsampled_output1, upsampled_output2], dim=dim)
        print(f"test_noisy_latent:{noisy_latents.shape}")
        return noisy_latents

def create_model_and_processing_logic(device):
    model = MyModel(device)
    process_latents = ProcessLatents(device)
    return model, process_latents

def process_noisy_latents(noisy_latent, model, process_latents, device, is_for_height=False):
    output = model(noisy_latent)
    print("Output shape after initial layers:", output.shape)

    output = process_latents(noisy_latent, device, is_for_height=is_for_height)
    print("Output shape after processing latents:", output.shape)

