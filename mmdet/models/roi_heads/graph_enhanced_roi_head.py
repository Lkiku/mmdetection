from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList
from mmdet.structures import DetDataSample, SampleList
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from .graph_modules import build_graph_from_proposals, GraphEmbedding, augment_graph
import torch.nn.functional as F

from mmdet.structures.bbox import bbox2roi
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances

@MODELS.register_module()
class GraphEnhancedRoIHead(StandardRoIHead):
    """Graph Enhanced RoI Head for object detection.
    
    This RoI head enhances the standard RoI head by incorporating graph-based
    relationship modeling between proposals.
    
    Args:
        graph_embedding_dim (int): Dimension of graph embeddings. Default: 128
        graph_hidden_dim (int): Hidden dimension in graph neural networks. Default: 256
        enable_graph_aug (bool): Whether to enable graph augmentation. Default: True
        graph_dropout (float): Dropout rate in graph neural networks. Default: 0.1
        **kwargs: Arguments passed to parent class
    """
    
    def __init__(self,
                 graph_embedding_dim: int = 256,
                 graph_hidden_dim: int = 256,
                 enable_graph_aug: bool = True,
                 graph_dropout: float = 0.1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.graph_embedding_dim = graph_embedding_dim
        self.enable_graph_aug = enable_graph_aug
        
        # Initialize graph embedding module
        self.graph_embedding_layer = GraphEmbedding(
            in_dim=256,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_embedding_dim,
            dropout=graph_dropout
        )

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection roi on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            # 添加视角一致性损失
            if 'loss_view_consistency' in bbox_results:
                losses['loss_view_consistency'] = bbox_results['loss_view_consistency']
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                        bbox_results['bbox_feats'],
                                        batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def bbox_loss(self, x: Tuple[Tensor],
                 sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict: Usually returns a dictionary with keys:
                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
                - `loss_view_consistency` (Tensor): View consistency loss if enabled.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def view_consistency_loss(self, graph_embeddings_1: torch.Tensor,
                            graph_embeddings_2: torch.Tensor, temperature=0.1) -> torch.Tensor:
        """Calculate view consistency loss between two graph embeddings.
        
        Args:
            graph_embeddings_1 (torch.Tensor): Graph embeddings from view 1
            graph_embeddings_2 (torch.Tensor): Graph embeddings from view 2
            temperature (float): Softmax temperature
            
        Returns:
            torch.Tensor: Consistency loss value
        """
        # cos_sim = F.cosine_similarity(graph_embeddings_1, graph_embeddings_2, dim=-1)
        # return torch.mean(1 - cos_sim)  # Maximize similarity
        # 先对特征进行L2归一化，但保持维度不变
        norm_1 = torch.norm(graph_embeddings_1, p=2, dim=-1, keepdim=True)
        norm_2 = torch.norm(graph_embeddings_2, p=2, dim=-1, keepdim=True)
        
        graph_embeddings_1_norm = graph_embeddings_1 / (norm_1 + 1e-7)
        graph_embeddings_2_norm = graph_embeddings_2 / (norm_2 + 1e-7)
        
        # 计算余弦相似度，添加数值稳定性
        cos_sim = F.cosine_similarity(graph_embeddings_1_norm, graph_embeddings_2_norm, dim=-1)
        
        # 使用 smooth L1 loss 来减少异常值的影响
        loss = F.smooth_l1_loss(cos_sim, torch.ones_like(cos_sim), reduction='none')
        
        # 添加动态权重：对于相似度较低的样本赋予更高的权重
        weights = (1 - cos_sim).detach()  # 停止梯度传播
        weights = F.softmax(weights / temperature, dim=0)  # 使用 temperature 控制权重分布
        
        # 计算加权平均损失
        weighted_loss = (loss * weights).sum()
        
        return weighted_loss

    
    def extract_roi_features(self,
                           x: Tuple[torch.Tensor],
                           proposals: List[torch.Tensor]) -> torch.Tensor:
        """Extract RoI features from image features.
        
        Args:
            x (tuple[Tensor]): List of multi-level img features.
            proposals (list[Tensor]): List of region proposals.
            
        Returns:
            Tensor: RoI features of shape (N, C, H, W).
        """
        if not hasattr(self, 'roi_extractor'):
            raise AttributeError("ROI extractor is not initialized")
            
        rois = proposals
        bbox_feats = self.roi_extractor(x[:self.roi_extractor.num_inputs], rois)
        
        return bbox_feats

    def simple_test(self,
                   x: Tuple[torch.Tensor],
                   proposals: List[torch.Tensor],
                   **kwargs) -> List[torch.Tensor]:
        """Test without augmentation.
        
        Args:
            x (tuple[Tensor]): Features from upstream network.
            proposals (list[Tensor]): Proposals from rpn head.
            
        Returns:
            list[Tensor]: Detection results.
        """
        # Extract features and enhance with graph information
        bbox_feats = self.extract_roi_features(x, proposals)
        cls_scores = self.bbox_head.forward_cls(bbox_feats)
        
        # Process each image in the batch
        enhanced_feats = []
        for i, (props, feats, scores) in enumerate(zip(proposals, bbox_feats, cls_scores)):
            # Build and process graph
            graph = build_graph_from_proposals(props, scores, feats)
            
            try:
                graph_embeddings = self.graph_embedding_layer(
                    graph['nodes'],
                    graph['edges']
                )
                enhanced_feat = torch.cat([feats, graph_embeddings], dim=1)
                enhanced_feats.append(enhanced_feat)
            except Exception as e:
                print(f"Warning: Graph processing failed for test sample {i}: {str(e)}")
                enhanced_feats.append(feats)  # Use original features as fallback
        
        # Stack enhanced features
        enhanced_feats = torch.stack(enhanced_feats)
        
        # Forward through bbox head
        results = self.bbox_head.simple_test(
            enhanced_feats,
            proposals,
            **kwargs
        )
        
        return results

    def _bbox_forward(self, x, rois):
        """Forward function for bbox head."""
        # 1. 提取ROI特征
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)  # [N, 256, 7, 7]
        
        # 2. 构建和处理图
        N = bbox_feats.size(0)
        
        # 检查rois的格式
        if rois.size(1) != 4:
            rois = rois[:, 1:]
        
        # 2. 获取初始预测 - 避免不必要的复制
        if self.bbox_head.with_avg_pool:
            pooled_feats = self.bbox_head.avg_pool(bbox_feats)
            temp_feats = pooled_feats.flatten(1)
        else:
            temp_feats = bbox_feats.flatten(1)  # [N, 256*7*7]
            # 使用expand替代cat，节省内存
            temp_feats = torch.cat([temp_feats, temp_feats.detach()], dim=1)  # [N, 2*256*7*7]
            
        # 通过共享FC层获取初始预测
        if hasattr(self.bbox_head, 'shared_fcs'):
            for fc in self.bbox_head.shared_fcs:
                temp_feats = self.bbox_head.relu(fc(temp_feats))
        
        # 使用torch.no_grad()避免存储不必要的梯度
        with torch.no_grad():
            init_cls_score = self.bbox_head.fc_cls(temp_feats)
            init_cls_probs = init_cls_score.softmax(dim=1)
        
        del temp_feats  # 释放内存
        
        consistency_loss = None
        try:
            # 视角1：原始图
            graph_1 = build_graph_from_proposals(
                proposals=rois,
                cls_scores=init_cls_probs,  # 直接使用概率
                bbox_features=bbox_feats
            )
            
            # 视角2：增强后的图
            if self.enable_graph_aug and self.training:
                graph_2 = augment_graph(graph_1.copy())
                
                # 获取两个视角的图嵌入
                graph_embeddings_1 = self.graph_embedding_layer(
                    graph_1['nodes'], 
                    graph_1['edges'],
                    graph_1['edge_weights']
                )
                graph_embeddings_2 = self.graph_embedding_layer(
                    graph_2['nodes'],
                    graph_2['edges'],
                    graph_2['edge_weights']
                )
                
                # 计算一致性损失
                consistency_loss = self.view_consistency_loss(
                    graph_embeddings_1,
                    graph_embeddings_2
                )
                
                # 使用原始视角的特征
                graph_embeddings = graph_embeddings_1
                
                # 清理不需要的图
                del graph_2
            else:
                # 测试时只使用原始图
                graph_embeddings = self.graph_embedding_layer(
                    graph_1['nodes'],
                    graph_1['edges'],
                    graph_1['edge_weights']
                )
            
            # 调整维度并扩展 - 使用view和expand避免复制
            graph_embeddings = graph_embeddings.view(N, -1, 1, 1).expand(-1, -1, 7, 7)
            
            # 合并特征 - 保持4D格式
            enhanced_feats = torch.cat([bbox_feats, graph_embeddings], dim=1)
            
            del graph_embeddings  # 释放内存
            
        except Exception as e:
            if self.training:
                raise e
            print(f"Warning: Graph processing failed in inference: {str(e)}")
            enhanced_feats = bbox_feats
        
        # 3. 最终预测
        if self.bbox_head.with_avg_pool:
            enhanced_feats = self.bbox_head.avg_pool(enhanced_feats)  # [N, C+graph_dim, 1, 1]
            enhanced_feats = enhanced_feats.flatten(1)  # [N, C+graph_dim]
        else:
            enhanced_feats = enhanced_feats.flatten(1)  # [N, (C+graph_dim)*7*7]
            
        if hasattr(self.bbox_head, 'shared_fcs'):
            for fc in self.bbox_head.shared_fcs:
                enhanced_feats = self.bbox_head.relu(fc(enhanced_feats))
                
        cls_score = self.bbox_head.fc_cls(enhanced_feats)
        bbox_pred = self.bbox_head.fc_reg(enhanced_feats)
        
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=enhanced_feats
        )

        if self.training and consistency_loss is not None:
            bbox_results['loss_view_consistency'] = consistency_loss * 1.0

        return bbox_results
