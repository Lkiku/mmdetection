import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SemanticAffinity(nn.Module):
    """语义亲和度计算模块"""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, features1, features2):
        """计算两组特征之间的语义亲和度
        
        Args:
            features1 (torch.Tensor): 第一组特征 [N, D]
            features2 (torch.Tensor): 第二组特征 [M, D]
            
        Returns:
            torch.Tensor: 亲和度矩阵 [N, M]
        """
        feat1 = self.feature_transform(features1)
        feat2 = self.feature_transform(features2)
        
        # 使用归一化和矩阵乘法优化余弦相似度计算
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        affinity = torch.mm(feat1_norm, feat2_norm.t())
        
        return affinity

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 使用单个线性层替代分别的Q,K,V变换
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 使用reshape和permute优化维度变换
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用scaled_dot_product优化注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

def add_scale_awareness(proposals, features):
    """添加尺度感知
    Args:
        proposals: [N, 4] 检测框
        features: List of FPN features
    """
    areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    scales = torch.sqrt(areas)
    
    # 将检测框分配到不同的FPN层级
    scale_ranges = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, float('inf')]
    ]
    
    level_assignments = torch.zeros(len(proposals), dtype=torch.long)
    for level, (l, h) in enumerate(scale_ranges):
        mask = (scales >= l) & (scales < h)
        level_assignments[mask] = level
        
    return level_assignments


def select_topk_nodes(nodes, k, by='score'):
    """选择每个类别的Top-K个节点
    Args:
        nodes: List of node dicts
        k: 每个类别保留的节点数
        by: 排序依据
    """
    scores = torch.tensor([n[by] for n in nodes])
    cls_dists = torch.stack([n['cls_dist'] for n in nodes])
    classes = cls_dists.argmax(dim=1)
    
    selected_nodes = []
    for c in range(cls_dists.size(1)):
        class_mask = (classes == c)
        if not class_mask.any():
            continue
            
        class_scores = scores[class_mask]
        class_indices = class_mask.nonzero().squeeze(1)
        
        if len(class_scores) > k:
            _, top_k = class_scores.topk(k)
            selected_indices = class_indices[top_k]
            selected_nodes.extend([nodes[i] for i in selected_indices])
            
    return selected_nodes



def build_graph_from_proposals(proposals, cls_scores, bbox_features, fpn_features=None, num_nodes_per_class=0):
    """构建基于proposal的图结构，使用增强的语义和空间关系
    
    Args:
        proposals (torch.Tensor): Shape (N, 4), 检测框坐标 [x1,y1,x2,y2]
        cls_scores (torch.Tensor): Shape (N, num_classes), 类别预测分数
        bbox_features (torch.Tensor): Shape (N, C, H, W), 检测框特征
        fpn_features (List[torch.Tensor], optional): FPN特征列表，用于尺度感知
        num_nodes_per_class (int, optional): 每个类别保留的最大节点数，0表示保留所有节点
    
    Returns:
        dict: 包含节点和边信息的图结构
    """
    device = proposals.device
    
    # 如果proposals是[N, 5]格式，取后4列
    if proposals.size(1) == 5:
        proposals = proposals[:, 1:]
    
    # 确保proposals现在是[N, 4]
    assert proposals.size(1) == 4, f"Expected proposals shape [N, 4], got {proposals.shape}"
    
    N = proposals.size(0)
    
    # 计算中心点
    x_center = (proposals[:, 0] + proposals[:, 2]).unsqueeze(1) / 2
    y_center = (proposals[:, 1] + proposals[:, 3]).unsqueeze(1) / 2
    centers = torch.cat([x_center, y_center], dim=1)  # [N, 2]
    
    # 将特征展平为 (N, -1)
    if bbox_features.dim() == 4:
        bbox_features = bbox_features.view(N, -1)
    
    # 计算检测框的宽高
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    areas = widths * heights
    box_sizes = torch.sqrt(areas)
    
    # 计算IoU矩阵
    def box_iou(box1, box2):
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou
    
    iou_matrix = box_iou(proposals, proposals)

    # 计算语义相似度
    cls_similarity = F.cosine_similarity(cls_scores.unsqueeze(1), 
                                       cls_scores.unsqueeze(0), dim=2)
    
    # 计算空间距离
    spatial_dist = torch.cdist(centers, centers)
    max_dist = spatial_dist.max()
    spatial_affinity = 1 - (spatial_dist / max_dist)
            
    # 综合多个关系计算最终的边权重
    edge_weights = (0.4 * iou_matrix + 
                   0.3 * cls_similarity + 
                   0.3 * spatial_affinity)
    
    # 使用自适应阈值选择边
    edge_threshold = edge_weights.mean() + edge_weights.std()
    valid_edges = (edge_weights > edge_threshold) & (edge_weights < 1.0)  # 排除自环
    edges = valid_edges.nonzero(as_tuple=False).tolist()
    edge_weights = edge_weights[valid_edges]  # 只保留有效边的权重
    
    # 预计算所有分数的最大值
    cls_max_scores = cls_scores.max(dim=1)[0]  # [N]
    
    # 构建节点特征
    nodes = [{
        'center': centers[i],
        'feature': bbox_features[i],
        'score': cls_max_scores[i],
        'cls_dist': cls_scores[i],
        'size': box_sizes[i],
        'area': areas[i]
    } for i in range(N)]
    
    # 添加尺度感知
    if fpn_features is not None:
        level_assignments = add_scale_awareness(proposals, fpn_features)
        for i, node in enumerate(nodes):
            node['level'] = level_assignments[i]
    
    # 根据类别选择Top-K个节点
    if num_nodes_per_class > 0:
        selected_nodes = select_topk_nodes(nodes, k=num_nodes_per_class)
        
        # 创建选中节点的索引映射
        selected_indices = {i: new_i for new_i, node in enumerate(selected_nodes) 
                          for i, orig_node in enumerate(nodes) if node is orig_node}
        
        # 更新边和权重
        new_edges = []
        new_edge_weights = []
        for edge_idx, (i, j) in enumerate(edges):
            if i in selected_indices and j in selected_indices:
                new_edges.append([selected_indices[i], selected_indices[j]])
                new_edge_weights.append(edge_weights[edge_idx])
        
        # 更新图结构
        nodes = selected_nodes
        edges = new_edges
        edge_weights = torch.tensor(new_edge_weights, device=device)
    
        # 更新距离矩阵
        N_new = len(nodes)
        new_spatial_dist = torch.zeros((N_new, N_new), device=device)
        new_cls_similarity = torch.zeros((N_new, N_new), device=device)
        for i, old_i in selected_indices.items():
            for j, old_j in selected_indices.items():
                new_spatial_dist[old_i, old_j] = spatial_dist[i, j]
                new_cls_similarity[old_i, old_j] = cls_similarity[i, j]
        spatial_dist = new_spatial_dist
        cls_similarity = new_cls_similarity
    
    return {
        'nodes': nodes,
        'edges': edges,
        'edge_weights': edge_weights if isinstance(edge_weights, torch.Tensor) else torch.tensor(edge_weights, device=device),
        'spatial_dist': spatial_dist,
        'cls_similarity': cls_similarity
    }


def augment_graph(graph, 
                 rotation_angle_range=(-15, 15),
                 scale_range=(0.9, 1.1),
                 translation_ratio=0.1,
                 feature_noise_std=0.01,
                 edge_weight_noise_std=0.01):
    """增强图结构，包括几何变换和特征扰动，但保持图的拓扑结构不变
    
    Args:
        graph (dict): 包含nodes和edges的图结构
        rotation_angle_range (tuple): 旋转角度范围(度)
        scale_range (tuple): 尺度变化范围
        translation_ratio (float): 平移比例
        feature_noise_std (float): 特征噪声的标准差
        edge_weight_noise_std (float): 边权重噪声的标准差
        
    Returns:
        dict: 增强后的图结构
    """
    nodes = graph['nodes']
    edges = graph['edges'].copy()
    spatial_dist = graph['spatial_dist'].clone()
    edge_weights = graph['edge_weights'].clone() if 'edge_weights' in graph else None
    cls_similarity = graph['cls_similarity'].clone() if 'cls_similarity' in graph else None
    
    N = len(nodes)
    
    if N == 0:
        return graph
    
    # 获取设备信息
    device = nodes[0]['center'].device
    
    # 获取所有中心点坐标
    centers = torch.stack([n['center'] for n in nodes])  # [N, 2]
    
    # 1. 几何变换增强 - 同时应用多个变换
    # 1.1 旋转变换
    angle = torch.empty(1, device=device).uniform_(float(rotation_angle_range[0]), float(rotation_angle_range[1]))
    angle_rad = angle * torch.pi / 180
    rot_mat = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                          [torch.sin(angle_rad), torch.cos(angle_rad)]], device=device)
    centers = torch.matmul(centers, rot_mat.t())
    
    # 1.2 缩放变换
    scale = torch.empty(1, device=device).uniform_(float(scale_range[0]), float(scale_range[1]))
    centers = centers * scale
    
    # 1.3 平移变换
    max_translation = centers.abs().max(dim=0)[0]  # [2]
    translation = torch.empty(2, device=device).uniform_(-1, 1) * max_translation * translation_ratio
    centers = centers + translation
    
    # 2. 特征增强 - 保持4D特征格式
    if feature_noise_std > 0:
        for node in nodes:
            # 确保特征保持4D格式 [C, H, W]
            feature = node['feature']  # [C, H, W]
            if feature.dim() == 3:
                feature_noise = torch.randn_like(feature) * feature_noise_std
                feature = feature + feature_noise
                node['feature'] = feature
    
    # 3. 边权重增强
    if edge_weight_noise_std > 0 and len(edges) > 0:
        # 为边权重添加噪声，但保持对称性
        noise = torch.randn_like(spatial_dist) * edge_weight_noise_std
        noise = (noise + noise.t()) / 2  # 确保噪声矩阵是对称的
        spatial_dist = spatial_dist + noise
        
        # 确保权重非负
        spatial_dist = torch.clamp(spatial_dist, min=0.0)
        
        if edge_weights is not None:
            edge_noise = torch.randn_like(edge_weights) * edge_weight_noise_std
            edge_weights = torch.clamp(edge_weights + edge_noise, min=0.0)
    
    # 4. 更新节点位置
    for i, center in enumerate(centers):
        nodes[i]['center'] = center
    
    # 5. 重新计算空间距离
    if len(edges) > 0:
        spatial_dist = torch.cdist(centers, centers)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'spatial_dist': spatial_dist,
        'edge_weights': edge_weights if edge_weights is not None else None,
        'cls_similarity': cls_similarity if cls_similarity is not None else None
    }


class GraphEmbedding(nn.Module):
    """增强的图嵌入模块，使用多头注意力和残差连接"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # 使用1x1卷积进行特征降维
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 多头注意力层
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # 图卷积层
        self.gcn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.gcn2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 残差连接的投影层
        self.shortcut = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        
    def forward(self, nodes, edges, edge_weights=None):
        """前向传播
        
        Args:
            nodes (list): 节点列表
            edges (list): 边列表
            edge_weights (torch.Tensor, optional): 边权重
            
        Returns:
            torch.Tensor: 图嵌入特征
        """
        device = nodes[0]['feature'].device
        num_nodes = len(nodes)
        
        # 准备节点特征
        node_feats = torch.stack([n['feature'] for n in nodes])  # [N, C] or [N, C, H, W]
        
        # 如果输入是2D的，重塑为4D
        if node_feats.dim() == 2:
            N, C = node_feats.shape
            # 确保通道数匹配预期的输入维度
            node_feats = node_feats.view(N, self.in_dim, int(math.sqrt(C // self.in_dim)), -1)
        
        # 使用1x1卷积降维
        node_feats = self.dim_reduce(node_feats)  # [N, hidden_dim, H, W]
        
        # 展平特征用于后续处理
        N = node_feats.size(0)
        node_feats = node_feats.view(N, self.hidden_dim, -1).mean(-1)  # [N, hidden_dim]
        
        # 构建邻接矩阵
        adj_matrix = self.build_adjacency_matrix(num_nodes, edges, device, edge_weights)
        
        # 自注意力处理
        node_feats = node_feats.unsqueeze(0)  # 添加batch维度
        attn_feats = self.self_attn(node_feats)
        node_feats = attn_feats.squeeze(0)  # 移除batch维度
        
        # 第一个图卷积层
        gcn1_feats = self.gcn1(torch.sparse.mm(adj_matrix, node_feats))
        
        # 第二个图卷积层（带残差连接）
        gcn2_feats = self.gcn2(torch.sparse.mm(adj_matrix, gcn1_feats))
        out_feats = gcn2_feats + self.shortcut(gcn1_feats)
        
        return out_feats
    
    @staticmethod
    def build_adjacency_matrix(num_nodes, edges, device, edge_weights=None):
        """构建稀疏格式的归一化邻接矩阵"""
        if not edges:
            # 如果没有边，返回单位矩阵
            return torch.eye(num_nodes, device=device).to_sparse()
            
        # 构建稀疏邻接矩阵
        indices = torch.tensor(edges, device=device).t()
        if edge_weights is not None:
            values = edge_weights.to(device)  # 确保在正确的设备上
        else:
            values = torch.ones(indices.size(1), device=device)
            
        # 确保对称性
        indices = torch.cat([indices, indices.flip(0)], dim=1)
        values = torch.cat([values, values], dim=0)
        
        # 创建稀疏张量
        adj = torch.sparse_coo_tensor(
            indices=indices, 
            values=values,
            size=(num_nodes, num_nodes),
            device=device
        ).coalesce()  # 合并重复索引并求平均
        
        # 计算度矩阵
        degrees = torch.sparse.sum(adj, dim=1).to_dense()
        degree_sqrt_inv = torch.pow(degrees + 1e-6, -0.5)
        
        # 创建对角度矩阵
        D_indices = torch.arange(num_nodes, device=device)
        D_indices = torch.stack([D_indices, D_indices], dim=0)
        D_sqrt_inv = torch.sparse_coo_tensor(
            indices=D_indices,
            values=degree_sqrt_inv,
            size=(num_nodes, num_nodes),
            device=device
        )
        
        # 归一化邻接矩阵 (使用稀疏矩阵运算)
        norm_adj = torch.sparse.mm(torch.sparse.mm(D_sqrt_inv, adj), D_sqrt_inv)
        
        return norm_adj