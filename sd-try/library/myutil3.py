import torch
import torch.nn as nn
import math

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model).to(device)
        self.wk = nn.Linear(d_model, d_model).to(device)
        self.wv = nn.Linear(d_model, d_model).to(device)
        self.dense = nn.Linear(d_model, d_model).to(device)

    def split_heads(self, x, batch_size):
        """分割最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q):
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)

        # Scaled Dot-Product Attention
        scaled_attention, _ = scaled_dot_product_attention(q, k, v)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(original_size_attention)

        return output

def scaled_dot_product_attention(q, k, v):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    dk = q.size()[-1]
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights

class AdaptiveLoss(nn.Module):
    def __init__(self, d_model, num_heads, huber_c=0.3):
        super(AdaptiveLoss, self).__init__()
        self.huber_c = huber_c
        self.multihead_attn = MultiHeadAttention(d_model, num_heads)
        self.cov2half = conv_layer = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride = 2).to(device)
        self.linear = nn.Linear(d_model, 2).to(device)  # 输出两个权重，一个给Huber损失，一个给L2损失

    def huber_loss(self, output, target):
        huber_loss = 2 * self.huber_c * (torch.sqrt((output - target) ** 2 + self.huber_c**2) - self.huber_c)
        print(f"myutil—— huber_loss:{huber_loss.shape},max:{torch.max(huber_loss)},min:{torch.min(huber_loss)}")
        return self.cov2half(huber_loss)

    def l2_loss(self, output, target):
        l2_loss = torch.nn.functional.mse_loss(output, target, reduction='none')
        print(f"myutil—— l2_loss:{l2_loss.shape},max:{torch.max(l2_loss)},min:{torch.min(l2_loss)}")
        return self.cov2half(l2_loss)

    def forward(self, output, target):
        print(f"myutil——output:{output.shape},max:{torch.max(output)},min:{torch.min(output)}")
        huber_loss = self.huber_loss(output, target)
        l2_loss = self.l2_loss(output, target)
        huber_loss = huber_loss - torch.min(huber_loss)
        l2_loss =l2_loss - torch.min(l2_loss)
        print(f"myutil—— huber_loss:{huber_loss.shape},max:{torch.max(huber_loss)},min:{torch.min(huber_loss)}")
        print(f"myutil—— l2_loss:{l2_loss.shape},max:{torch.max(l2_loss)},min:{torch.min(l2_loss)}")
        combined_loss = torch.stack([huber_loss, l2_loss], dim=-1)  # 形状为[2,4,64,64,2]
        print(f"myutil—— combined_loss1:{combined_loss.shape},max:{torch.max(combined_loss)},min:{torch.min(combined_loss)}")
        batch_size, channels, height, width, _ = combined_loss.shape
        combined_loss = combined_loss.view(batch_size, -1, 2)  # 形状为[batch_size, seq_len, 2]
        print(f"myutil—— combined_loss2:{combined_loss.shape},max:{torch.max(combined_loss)},min:{torch.min(combined_loss)}")
        # 调整形状使其与d_model匹配
        if combined_loss.shape[-1] != self.multihead_attn.d_model:
            combined_loss = torch.cat([combined_loss] * (self.multihead_attn.d_model // combined_loss.shape[-1]), dim=-1)
            print(f"myutil—— combined_loss3:{combined_loss.shape},max:{torch.max(combined_loss)},min:{torch.min(combined_loss)}")
        attn_output = self.multihead_attn(combined_loss, combined_loss, combined_loss).to(device)
        
        print(f"myutil—— attn_output:{attn_output.shape},max:{torch.max(attn_output)},min:{torch.min(attn_output)}")
        attn_weights = self.linear(attn_output)  # 形状为[batch_size, seq_len, 2]
        print(f"myutil—— attn_weights:{attn_weights.shape},max:{torch.max(attn_weights)},min:{torch.min(attn_weights)}")
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        print(f"myutil—— attn_weights2:{attn_weights.shape},max:{torch.max(attn_weights)},min:{torch.min(attn_weights)}")
        huber_weight = attn_weights[..., 0].view(batch_size, channels, height, width)
        print(f"myutil—— huber_weight:{huber_weight.shape},max:{torch.max(huber_weight)},min:{torch.min(huber_weight)}")
        l2_weight = attn_weights[..., 1].view(batch_size, channels, height, width)
        print(f"myutil—— l2_weight:{l2_weight.shape},max:{torch.max(l2_weight)},min:{torch.min(l2_weight)}")
        final_loss = huber_weight * huber_loss + l2_weight * l2_loss
        print(f"myutil—— final_loss:{final_loss.shape},max:{torch.max(final_loss)},min:{torch.min(final_loss)}")
        return final_loss

def create_loss_weight(d_model=128, num_heads=8, huber_c=0.3):
    adaptive_loss_fn = AdaptiveLoss(d_model=d_model, num_heads=num_heads, huber_c=huber_c)
    return adaptive_loss_fn

# 计算动态加权损失
def compute_dynamic_weights(weight_loss_fn, output, target):
    return weight_loss_fn(output, target)
