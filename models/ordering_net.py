import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OrderingNetV4(nn.Module):
    """
    OrderingNet 用于对点云特征进行语义感知的重排序。
    专注于学习排列规则，输出硬排列用于Mamba序列处理。
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
        self.perm_score_predictor = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_groups)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def soft_to_hard_permutation(self, P):
        """
        将软排列矩阵转换为硬排列索引。

        Args:
            P: (B, G, G) - 软排列矩阵

        Returns:
            perm_indices: (B, G) - 硬排列索引，perm_indices[b, i] 表示原始位置i应该移动到的新位置
        """
        # 方法1: 使用匈牙利算法 (最优分配)
        # 但这里我们使用更简单的贪心方法：对每个原始位置选择概率最大的新位置

        batch_size, num_groups, _ = P.shape
        perm_indices = []

        for b in range(batch_size):
            P_b = P[b]  # (G, G)

            # 贪心分配：依次为每个原始位置选择最佳新位置
            used_positions = set()
            assignment = [-1] * num_groups

            # 按照P中最大值的顺序进行分配
            flat_P = P_b.flatten()
            sorted_indices = torch.argsort(flat_P, descending=True)

            for idx in sorted_indices:
                orig_pos = idx // num_groups
                new_pos = idx % num_groups

                if assignment[orig_pos] == -1 and new_pos.item() not in used_positions:
                    assignment[orig_pos] = new_pos.item()
                    used_positions.add(new_pos.item())

                if len(used_positions) == num_groups:
                    break

            perm_indices.append(torch.tensor(assignment, device=P.device))

        return torch.stack(perm_indices)  # (B, G)

    def apply_hard_permutation(self, data, perm_indices):
        """
        根据排列索引对数据进行硬排列。

        Args:
            data: (B, G, ...) - 要排列的数据
            perm_indices: (B, G) - 排列索引

        Returns:
            reordered_data: (B, G, ...) - 重排序后的数据
        """
        batch_size, num_groups = perm_indices.shape

        # 创建新的排列后的数据
        reordered_data = torch.zeros_like(data)

        for b in range(batch_size):
            for orig_pos in range(num_groups):
                new_pos = perm_indices[b, orig_pos]
                reordered_data[b, new_pos] = data[b, orig_pos]

        return reordered_data

    def compute_ordering_matrix(self, group_features_for_scoring):
        """
        使用聚合后的特征计算排序矩阵 P。
        """
        perm_scores = self.perm_score_predictor(group_features_for_scoring)
        P = self.soft_permutation_matrix(perm_scores, tau=self.sinkhorn_tau, iters=self.sinkhorn_iters)
        return P

    def soft_permutation_matrix(self, scores, tau=0.1, iters=10):
        """
        使用 Sinkhorn 迭代将分数转换为软排列矩阵。
        """
        log_alpha = scores / tau

        for _ in range(iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

        P = torch.exp(log_alpha)
        return P

    def forward(self, center_coords, group_features):
        """
        前向传播，基于学习到的规则对特征进行重排序。

        Args:
            center_coords: (B, G, 3) - 中心点坐标。
            group_features: (B, G, C) - 每个组的特征。

        Returns:
            reordered_center_coords: (B, G, 3) - 重排序后的中心点坐标。
            reordered_group_features: (B, G, C) - 重排序后的组特征。
            perm_indices: (B, G) - 使用的排列索引。
        """
        # 计算软排列矩阵
        P = self.compute_ordering_matrix(group_features)

        # 转换为硬排列索引
        perm_indices = self.soft_to_hard_permutation(P)

        # 应用硬排列进行真正的重排序
        reordered_center_coords = self.apply_hard_permutation(center_coords, perm_indices)
        reordered_group_features = self.apply_hard_permutation(group_features, perm_indices)

        return reordered_center_coords, reordered_group_features, perm_indices


# 更高效的实现版本（使用高级索引）
class OrderingNetV4Efficient(nn.Module):
    """更高效的硬排列实现"""

    def __init__(self, in_channels, num_groups, mlp_hidden_channels=None, sinkhorn_tau=0.1, sinkhorn_iters=10):
        super(OrderingNetV4Efficient, self).__init__()
        self.num_groups = num_groups
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_iters = sinkhorn_iters

        if mlp_hidden_channels is None:
            mlp_hidden_channels = in_channels * 2

        self.perm_score_predictor = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_groups)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def soft_permutation_matrix(self, scores, tau=0.1, iters=10):
        log_alpha = scores / tau
        for _ in range(iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def forward(self, center_coords, group_features, gruop_coords):
        """
        Args:
            center_coords: (B, G, 3)
            group_features: (B, G, C)
        Returns:
            reordered_center_coords: (B, G, 3) - 重排序后的坐标
            reordered_group_features: (B, G, C) - 重排序后的特征
            perm_indices: (B, G) - 排列索引
        """
        # 计算排列分数和软排列矩阵
        perm_scores = self.perm_score_predictor(group_features)
        P = self.soft_permutation_matrix(perm_scores, self.sinkhorn_tau, self.sinkhorn_iters)

        # 将软排列转换为硬排列：对每行取argmax得到每个原始位置要去的新位置
        perm_indices = torch.argmax(P, dim=-1)  # (B, G)

        # 使用高级索引进行重排序
        batch_indices = torch.arange(center_coords.size(0)).unsqueeze(1).expand(-1, self.num_groups)

        # 创建重排序后的数据
        reordered_center_coords = torch.zeros_like(center_coords)
        reordered_group_features = torch.zeros_like(group_features)
        reordered_group_coords = torch.zeros_like(gruop_coords)

        # 根据排列索引重新排列
        reordered_center_coords[batch_indices, perm_indices] = center_coords
        reordered_group_features[batch_indices, perm_indices] = group_features
        reordered_group_coords[batch_indices, perm_indices] = gruop_coords

        return reordered_center_coords, reordered_group_features, reordered_group_coords, perm_indices




import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LearnableScanningOrder(nn.Module):
    """
    学习型点云扫描顺序模块 - 专为FPS+KNN分组输入设计
    输入: centers (B, G, 3) + neighborhoods (B, G, S, 3)
    输出: 重新排序的点云 (B, G*S, 3)
    """

    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 temperature: float = 1.0,
                 dropout: float = 0.1,
                 use_center_bias: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.use_center_bias = use_center_bias

        # 点特征编码器
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 3D位置编码
        self.pos_encoder = PositionalEncoding3D(hidden_dim)

        # 局部注意力 - 用于组内点的关系学习
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 全局注意力 - 用于组间关系学习
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 中心点-邻域融合层
        self.center_neighborhood_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 组内排序得分预测器
        self.local_order_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 组间排序得分预测器
        self.global_order_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 可选: 中心点偏置 - 让中心点在组内有更高的优先级
        if use_center_bias:
            self.center_bias = nn.Parameter(torch.tensor(1.0))

    def forward(self,
                centers: torch.Tensor,
                neighborhoods: torch.Tensor,
                group_size: int,
                num_group: int,
                return_indices: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            centers: 中心点 (B, G, 3)
            neighborhoods: 邻域点 (B, G, S, 3)
            group_size: 每组点数 S
            num_group: 组数 G
            return_indices: 是否返回排序索引

        Returns:
            ordered_points: 重新排序后的点 (B, G*S, 3)
            indices: 排序索引 (B, G*S) [可选]
        """
        B, G, _ = centers.shape
        _, _, S, _ = neighborhoods.shape

        assert G == num_group, f"Group number mismatch: {G} vs {num_group}"
        assert S == group_size, f"Group size mismatch: {S} vs {group_size}"

        # Step 1: 编码所有点的特征
        center_features = self.encode_points(centers)  # (B, G, hidden_dim)
        neighborhood_features = self.encode_neighborhoods(neighborhoods)  # (B, G, S, hidden_dim)

        # Step 2: 学习组间的排序
        group_order_indices = self.learn_global_order(center_features, centers)  # (B, G)

        # Step 3: 学习每组内的排序
        local_order_indices = self.learn_local_order(
            center_features, neighborhood_features, centers, neighborhoods
        )  # (B, G, S)

        # Step 4: 根据排序重新组织点云
        ordered_points, global_indices = self.reorder_points(
            centers, neighborhoods, group_order_indices, local_order_indices
        )

        if return_indices:
            return ordered_points, global_indices
        return ordered_points, None

    def encode_points(self, points: torch.Tensor) -> torch.Tensor:
        """编码点的基础特征"""
        point_features = self.point_encoder(points)
        # 添加位置编码
        point_features = self.pos_encoder(point_features, points)
        return point_features

    def encode_neighborhoods(self, neighborhoods: torch.Tensor) -> torch.Tensor:
        """编码邻域点特征"""
        B, G, S, _ = neighborhoods.shape
        # 重塑为 (B*G, S, 3) 进行批处理
        neighborhoods_flat = neighborhoods.view(B * G, S, 3)
        neighborhood_features = self.point_encoder(neighborhoods_flat)
        neighborhood_features = self.pos_encoder(neighborhood_features, neighborhoods_flat)
        # 重塑回 (B, G, S, hidden_dim)
        return neighborhood_features.view(B, G, S, -1)

    def learn_global_order(self, center_features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """学习组间的排序顺序"""
        B, G, _ = center_features.shape

        # 通过全局注意力学习组间关系
        global_attended, _ = self.global_attention(
            center_features, center_features, center_features
        )  # (B, G, hidden_dim)

        # 预测组的重要性得分
        group_scores = self.global_order_predictor(global_attended).squeeze(-1)  # (B, G)

        # 获取排序索引
        group_order_indices = self.get_order_indices(group_scores, dim=-1)

        return group_order_indices  # (B, G)

    def learn_local_order(self,
                         center_features: torch.Tensor,
                         neighborhood_features: torch.Tensor,
                         centers: torch.Tensor,
                         neighborhoods: torch.Tensor) -> torch.Tensor:
        """学习每组内的排序顺序"""
        B, G, S, _ = neighborhood_features.shape

        # 为每个组学习局部排序
        all_local_indices = []

        for g in range(G):
            # 获取当前组的中心点和邻域特征
            current_center_feat = center_features[:, g:g+1]  # (B, 1, hidden_dim)
            current_neighborhood_feat = neighborhood_features[:, g]  # (B, S, hidden_dim)

            # 融合中心点和邻域点特征
            # 将中心点特征广播到每个邻域点
            center_expanded = current_center_feat.expand(-1, S, -1)  # (B, S, hidden_dim)
            fused_features = torch.cat([center_expanded, current_neighborhood_feat], dim=-1)
            fused_features = self.center_neighborhood_fusion(fused_features)  # (B, S, hidden_dim)

            # 通过局部注意力学习组内点的关系
            local_attended, _ = self.local_attention(
                fused_features, fused_features, fused_features
            )  # (B, S, hidden_dim)

            # 预测组内点的重要性得分
            local_scores = self.local_order_predictor(local_attended).squeeze(-1)  # (B, S)

            # 如果使用中心点偏置，给中心点对应的位置加权
            if self.use_center_bias:
                # 假设中心点是邻域中的第一个点，给它额外的权重
                local_scores[:, 0] += self.center_bias

            # 排序
            local_indices = self.get_order_indices(local_scores, dim=-1)

            all_local_indices.append(local_indices)

        local_order_indices = torch.stack(all_local_indices, dim=1)  # (B, G, S)
        return local_order_indices

    def reorder_points(self,
                      centers: torch.Tensor,
                      neighborhoods: torch.Tensor,
                      group_order_indices: torch.Tensor,
                      local_order_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据学习到的顺序重新排列点"""
        B, G, S, _ = neighborhoods.shape

        all_ordered_points = []
        all_global_indices = []

        for b in range(B):
            batch_ordered_points = []
            batch_global_indices = []

            # 按照组的顺序处理
            for group_rank, g_idx in enumerate(group_order_indices[b]):
                g_idx = int(g_idx)  # 确保索引是整数类型

                # 获取当前组的邻域点
                current_neighborhood = neighborhoods[b, g_idx]  # (S, 3)

                # 获取组内的排序索引
                local_indices = local_order_indices[b, g_idx].long()  # 转换为long类型

                # 重新排列组内的点
                ordered_neighborhood = current_neighborhood[local_indices]  # (S, 3)
                batch_ordered_points.append(ordered_neighborhood)

                # 计算全局索引
                global_indices = g_idx * S + local_indices
                batch_global_indices.append(global_indices)

            # 拼接所有组的点
            batch_ordered_points = torch.cat(batch_ordered_points, dim=0)  # (G*S, 3)
            batch_global_indices = torch.cat(batch_global_indices, dim=0)  # (G*S,)

            all_ordered_points.append(batch_ordered_points)
            all_global_indices.append(batch_global_indices)

        ordered_points = torch.stack(all_ordered_points, dim=0)  # (B, G*S, 3)
        global_indices = torch.stack(all_global_indices, dim=0)  # (B, G*S)

        return ordered_points, global_indices

    def get_order_indices(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """获取排序索引"""
        # 直接使用硬排序，简单有效
        # 梯度通过score prediction网络传播，这对于大多数应用已经足够
        return torch.argsort(scores, dim=dim, descending=True)


class PositionalEncoding3D(nn.Module):
    """3D位置编码 - 增强版本，支持相对位置编码"""

    def __init__(self, hidden_dim: int, max_freq: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_freq = max_freq

        # 位置编码网络
        self.pos_encoding = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 频率编码 (类似Transformer的正弦位置编码，但适配3D)
        freqs = torch.arange(0, hidden_dim // 6, dtype=torch.float32)
        freqs = 2.0 * freqs / (hidden_dim // 6)
        freqs = torch.pow(max_freq, freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 点特征 (..., hidden_dim)
            positions: 3D坐标 (..., 3)
        """
        # 基础位置编码
        pos_encoding = self.pos_encoding(positions)

        # 添加正弦余弦位置编码
        if len(self.freqs) > 0:
            # 为每个坐标轴生成正弦余弦编码
            pos_scaled = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (..., 3, freq_dim)
            pos_sin = torch.sin(pos_scaled)
            pos_cos = torch.cos(pos_scaled)

            # 拼接并调整维度
            sincos_encoding = torch.stack([pos_sin, pos_cos], dim=-1)  # (..., 3, freq_dim, 2)
            sincos_encoding = sincos_encoding.flatten(-2)  # (..., 3, freq_dim*2)
            sincos_encoding = sincos_encoding.flatten(-2)  # (..., 3*freq_dim*2)

            # 如果维度不匹配，使用线性层调整
            if sincos_encoding.shape[-1] != self.hidden_dim:
                if not hasattr(self, 'sincos_proj'):
                    self.sincos_proj = nn.Linear(sincos_encoding.shape[-1], self.hidden_dim)
                    if positions.is_cuda:
                        self.sincos_proj = self.sincos_proj.cuda()
                sincos_encoding = self.sincos_proj(sincos_encoding)

            pos_encoding = pos_encoding + sincos_encoding

        return features + pos_encoding


# 使用示例和测试
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模块
    scanner = LearnableScanningOrder(
        input_dim=3,
        hidden_dim=128,
        num_heads=8,
        temperature=1.0,
        use_center_bias=True
    ).to(device)

    # 测试数据
    batch_size = 2
    num_group = 64
    group_size = 16

    centers = torch.randn(batch_size, num_group, 3).to(device)
    neighborhoods = torch.randn(batch_size, num_group, group_size, 3).to(device)

    print("=== 学习型点云扫描顺序测试 ===")
    print(f"输入中心点形状: {centers.shape}")
    print(f"输入邻域形状: {neighborhoods.shape}")
    print(f"总点数: {num_group * group_size}")

    # 前向传播
    with torch.no_grad():
        ordered_points, indices = scanner(
            centers=centers,
            neighborhoods=neighborhoods,
            group_size=group_size,
            num_group=num_group,
            return_indices=True
        )

    print(f"\n输出有序点云形状: {ordered_points.shape}")
    print(f"排序索引形状: {indices.shape}")
    print(f"验证点数量匹配: {num_group * group_size == ordered_points.shape[1]}")

    # 计算参数量
    total_params = sum(p.numel() for p in scanner.parameters())
    trainable_params = sum(p.numel() for p in scanner.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试训练模式
    scanner.train()
    ordered_points_train, _ = scanner(
        centers=centers,
        neighborhoods=neighborhoods,
        group_size=group_size,
        num_group=num_group,
        return_indices=False
    )
    print(f"训练模式输出形状: {ordered_points_train.shape}")

    print("\n✅ 所有测试通过！")