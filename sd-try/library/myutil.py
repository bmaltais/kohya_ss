import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads = 1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.query_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1).to(device)
        self.key_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1).to(device)
        self.value_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1).to(device)
        self.out_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1).to(device)
        self.gamma = nn.Parameter(torch.zeros(1)).to(device)

    def forward(self, x):
        print("Input shape:", x.shape)  # 输出输入数据的形状
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, self.hidden_channels // self.num_heads, self.num_heads, height * width).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, self.hidden_channels // self.num_heads, height * width)
        key = self.key_conv(x).view(batch_size, self.hidden_channels // self.num_heads, self.num_heads, height * width).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, self.hidden_channels // self.num_heads, height * width)
        value = self.value_conv(x).view(batch_size, self.hidden_channels // self.num_heads, self.num_heads, height * width).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, self.hidden_channels // self.num_heads, height * width)
        
        print("Query conv weight shape:", self.query_conv.weight.shape)  # 输出查询卷积层的权重形状
        print("Key conv weight shape:", self.key_conv.weight.shape)  # 输出键卷积层的权重形状
        print("Value conv weight shape:", self.value_conv.weight.shape)  # 输出值卷积层的权重形状
        energy = torch.bmm(query, key.transpose(1, 2))
        attention = torch.softmax(energy, dim=-1)
    
        attention_out = torch.bmm(attention, value)
        print("attention_out:", attention_out.shape)
        attention_out = attention_out.view(batch_size, self.num_heads, self.hidden_channels // self.num_heads, height * width).permute(0, 2, 1, 3).contiguous().view(batch_size, self.hidden_channels, height, width)
        print("attention_out:", attention_out.shape)
        attention_out = self.out_conv(attention_out)
        print("attention_out:", attention_out.shape)
        out = self.gamma * attention_out + x
        return out

class DynamicWeightedLoss(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_heads = 8):
        super(DynamicWeightedLoss, self).__init__()
        self.attention = SelfAttention(in_channels, hidden_channels,num_heads).to(device)
        self.fc = nn.Linear(hidden_channels, 1).to(device)

    def ssim_loss(self,target, pre_loss, window_size=11, sigma=1.5):
        # 获取数据范围
        min_val = torch.min(target).item()
        max_val = torch.max(target).item()
        data_range = max_val - min_val

        # 图像的均值、方差和协方差
        channels = target.shape[1]
        weight = torch.ones(channels, channels, window_size, window_size).to(target.device) / (window_size ** 2)
        mu1 = F.conv2d(target, weight, padding=window_size // 2)
        mu2 = F.conv2d(pre_loss, weight, padding=window_size // 2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        sigma1_sq = F.conv2d(target ** 2, weight, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(pre_loss ** 2, weight, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(target * pre_loss, weight, padding=window_size // 2) - mu12

        # SSIM计算
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        # SSIM损失
        ssim_loss = 1 - ssim_map

        return ssim_loss
    def forward(self, output, target, huber_c):
        # 确保输入张量也在 GPU 上
        output = output.to(device)
        target = target.to(device)
        huber_loss = 2 * huber_c * (torch.sqrt((output - target) ** 2 + huber_c**2) - huber_c)
        l2_loss = torch.nn.functional.mse_loss(output, target, reduction='none')
        #ssim_loss = self.ssim_loss(target, output)
        loss_values = torch.cat([huber_loss, l2_loss], dim=1)
        #loss_values = huber_loss
        print(f"myutil——loss_values:{loss_values.shape},max:{torch.max(loss_values)},min:{torch.min(loss_values)}")
        attention_out = self.attention(loss_values)
        print(f"myutil——1:{attention_out.shape},max:{torch.max(attention_out)},min:{torch.min(attention_out)}")
        weighted_loss = (attention_out * loss_values)
        print(f"myutil:weighted_loss{weighted_loss.shape},max:{torch.max(weighted_loss)},min:{torch.min(weighted_loss)}")
        return torch.sqrt(weighted_loss) + torch.cat([huber_loss, huber_loss], dim=1) * 0.5
        
def create_loss_weight(hidden_channels=64, in_channels=4,num_heads = 4):
    return DynamicWeightedLoss(in_channels=in_channels, hidden_channels=hidden_channels,num_heads = 4).to(device)

# 计算动态加权损失
def compute_dynamic_weights(weight_loss_fn, output, target, huber_c):
    return weight_loss_fn(output, target, huber_c)


