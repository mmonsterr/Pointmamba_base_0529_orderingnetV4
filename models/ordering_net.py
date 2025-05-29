import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OrderingNetV4(nn.Module):
    """
    OrderingNet 用于对点云特征进行语义感知的重排序。
    专注于学习排列规则，移除了显式的prompt学习。
    """

    def __init__(self, in_channels, num_groups, mlp_hidden_channels=None, sinkhorn_tau=0.1, sinkhorn_iters=10):
        """
        Args:
            in_channels (int): 输入特征的维度 (C)，对应你的 feat 的通道数。
            num_groups (int): 点云组的数量 (G)。
            mlp_hidden_channels (int, optional): 用于预测排列分数的MLP的隐藏层维度。
                                                默认为 in_channels * 2。
            sinkhorn_tau (float): Sinkhorn 迭代中用于计算 log_alpha 的温度参数。
            sinkhorn_iters (int): Sinkhorn 迭代的次数。
        """
        super(OrderingNetV4, self).__init__()
        self.num_groups = num_groups
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_iters = sinkhorn_iters

        if mlp_hidden_channels is None:
            mlp_hidden_channels = in_channels * 2

        # 网络用于直接从聚合特征预测排列分数
        # 输入 (B, G, C), 输出 (B, G, G)
        # perm_scores[b, i, j] 表示原始组 i 被映射到新位置 j 的分数
        self.perm_score_predictor = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_groups)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_ordering_matrix(self, group_features_for_scoring):
        """
        使用聚合后的特征计算排序矩阵 P。

        Args:
            group_features_for_scoring: (B, G, C) - 用于预测排列分数的特征。
        Returns:
            P: (B, G, G) - 排列矩阵。P[b, orig_idx, new_idx] 表示原始组 orig_idx 移动到新位置 new_idx 的概率/权重。
        """
        # B, G, C = group_features_for_scoring.shape # C 在这里未使用，但形状是这样的

        # 预测排列分数
        # perm_scores[b, i, j] 表示原始组 i (orig_idx) 被映射到新位置 j (new_idx) 的分数
        perm_scores = self.perm_score_predictor(group_features_for_scoring)  # (B, G_orig, G_new)

        # 使用 Sinkhorn 迭代将分数转换为 (软)排列矩阵
        P = self.soft_permutation_matrix(perm_scores, tau=self.sinkhorn_tau, iters=self.sinkhorn_iters)

        return P

    def soft_permutation_matrix(self, scores, tau=0.1, iters=10):
        """
        使用 Sinkhorn 迭代将分数转换为软排列矩阵。
        Args:
            scores: (B, G, G) - 分数矩阵。scores[b, i, j] 表示原始项 i 移动到新位置 j 的分数。
            tau: float - 温度参数。
            iters: int - Sinkhorn 迭代次数。
        Returns:
            P: (B, G, G) - 软排列矩阵。P[b, orig_idx, new_idx]
        """
        # B, G, _ = scores.shape # G 在这里未使用

        # 使用温度参数应用到分数上
        log_alpha = scores / tau

        # Sinkhorn 迭代得到双随机矩阵
        for _ in range(iters):
            # 按行归一化 (log_alpha[b, orig_idx, :]) - 确保每个原始项的概率分布和为1
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # 按列归一化 (log_alpha[b, :, new_idx]) - 确保每个新位置的概率分布和为1
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

        # 转换为概率
        P = torch.exp(log_alpha)

        return P

    def forward(self, center_coords, group_features):
        """
        前向传播，基于学习到的规则对特征进行重排序。

        Args:
            center_coords: (B, G, 3) - 中心点坐标。
            group_features: (B, G, C) - 每个组的特征，用于计算排序和被排序。

        Returns:
            reordered_center_coords: (B, G, 3) - 重排序后的中心点坐标。
            reordered_group_features: (B, G, C) - 重排序后的组特征。
            P: (B, G, G) - 使用的排列矩阵。
        """
        # group_features (B, G, C) 已经是从邻域点提取的特征，直接用于计算排序矩阵。
        # P[b, orig_idx, new_idx] 表示原始组 orig_idx 移动到新位置 new_idx 的概率/权重
        P = self.compute_ordering_matrix(group_features)

        # 应用排列来重排序特征
        # P: (B, G_orig, G_new)
        # center_coords: (B, G_orig, 3)
        # reordered_center_coords[b, new_idx, coord_dim] = sum_{orig_idx} P[b, orig_idx, new_idx] * center_coords[b, orig_idx, coord_dim]
        reordered_center_coords = torch.einsum('boc,boj->bjc', center_coords, P)

        # group_features: (B, G_orig, C)
        # reordered_group_features[b, new_idx, feat_dim] = sum_{orig_idx} P[b, orig_idx, new_idx] * group_features[b, orig_idx, feat_dim]
        reordered_group_features = torch.einsum('boc,boj->bjc', group_features, P)

        return reordered_center_coords, reordered_group_features, P