#多对1的代码，强行进行通道融合
# import torch
# from torch import nn
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.patch_size = 16  # 常见的 Patch 长度
#         self.stride = 8  # 步长
#         self.d_model = configs.d_model
#         self.n_heads = 8
#         self.e_layers = configs.e_layers
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in  # 这里的 enc_in 是你特征的总维度（比如16）
#
#         # 计算 Patch 后的序列长度
#         self.patch_num = (96 - self.patch_size) // self.stride + 1
#
#         # 【核心修正】：Embedding层现在接收 所有气象特征 × Patch长度 的信息
#         self.patch_embedding = nn.Linear(self.patch_size * self.enc_in, self.d_model)
#
#         # Transformer 编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=self.n_heads,
#             dim_feedforward=self.d_model * 4,
#             dropout=configs.dropout,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)
#
#         # 输出头
#         self.head = nn.Linear(self.d_model * self.patch_num, self.pred_len)
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # 现在的 x_enc 包含了所有气象特征，形状为 [Batch, 96, Features]
#
#         # 1. Patching: 沿着时间维度 (dim=1) 进行分块
#         # unfold 后形状变为 [Batch, patch_num, Features, patch_size]
#         x = x_enc.unfold(dimension=1, size=self.patch_size, step=self.stride)
#
#         # 2. 将气象特征维度和分块维度压平在一起
#         # 形状变为 [Batch, patch_num, Features * patch_size]
#         x = x.reshape(x.shape[0], self.patch_num, -1)
#
#         # 3. Embedding 映射
#         x = self.patch_embedding(x)  # [Batch, patch_num, d_model]
#
#         # 4. Transformer 全局注意力编码
#         x = self.encoder(x)  # [Batch, patch_num, d_model]
#
#         # 5. 展平并经过全连接层输出预测长度
#         x = x.reshape(x.shape[0], -1)  # [Batch, patch_num * d_model]
#         x = self.head(x)  # [Batch, 48]
#
#         # 增加一个维度，对齐最后的目标列输出 [Batch, 48, 1]
#         return x.unsqueeze(-1)

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.patch_size = 16  # Patch 长度
        self.stride = 8  # 滑动步长
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len

        # 计算分块数量: (96 - 16) // 8 + 1 = 11
        self.patch_num = (96 - self.patch_size) // self.stride + 1

        # Patch Embedding: 仅映射单个变量的 patch 长度 (16)
        self.patch_embedding = nn.Linear(self.patch_size, self.d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=configs.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)

        # 线性输出头
        self.head = nn.Linear(self.d_model * self.patch_num, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 此时的 x_enc 包含了所有变量，形状为: [Batch, 96, Features]

        # ========================================================
        # 【核心逻辑：通道独立 (Channel Independence)】
        # 强制剥离所有气象协变量，只保留最后一列（碳通量 NEE）进行自回归预测
        # ========================================================
        x = x_enc[:, :, -1:]  # 截取目标列 -> [Batch, 96, 1]
        x = x.transpose(1, 2)  # 转置适应分块 -> [Batch, 1, 96]

        # 1. 分块 (Patching)
        # unfold 后形状变为: [Batch, 1, patch_num, patch_size]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        # 2. 嵌入 (Embedding)
        x = self.patch_embedding(x)  # -> [Batch, 1, patch_num, d_model]
        x = x.squeeze(1)  # 压缩单变量通道维 -> [Batch, patch_num, d_model]

        # 3. 注意力编码 (Attention)
        x = self.encoder(x)  # -> [Batch, patch_num, d_model]

        # 4. 展平并预测 (Flatten & Linear Head)
        x = x.reshape(x.shape[0], -1)  # -> [Batch, patch_num * d_model]
        x = self.head(x)  # -> [Batch, 48]

        # 5. 增加一个维度，严格对齐主程序的真实标签维度 [Batch, 48, 1]
        return x.unsqueeze(-1)