import os.path as osp
import mmcv
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from mmengine.fileio import load
from mmengine.utils import mkdir_or_exist

import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a heart detector')
    parser.add_argument('--source', type=str, default='voch1', help='source domain for training')
    parser.add_argument('--target', type=str, default='voch2', help='target domain for testing')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to use')
    parser.add_argument('--suffix', type=str, default='', help='suffix to add to the experiment name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 注册所有模块
    register_all_modules()

    # 修改配置文件为 Faster R-CNN 101
    config_file = 'configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py'
    
    # 加载配置
    cfg = Config.fromfile(config_file)
    
    # 使用命令行参数设置数据域
    source_domain = args.source
    target_domain = args.target
    
    # 修改数据集路径
    cfg.dataset_type = 'CocoDataset'
    cfg.data_root = './data/coco_heart/'

    train_data_file = cfg.data_root + f'{source_domain}_train_560.json'
    test_data_file = cfg.data_root + f'{target_domain}_all_800.json'

    # 设置类别数量 - 使用源域的标注文件
    annotations = load(train_data_file)
    categories = annotations['categories']
    num_classes = len(categories)
    
    # 修改所有需要设置类别数量的地方
    cfg.model.roi_head.bbox_head.num_classes = num_classes
    # 从 annotations 中获取类别名称
    class_names = [category['name'] for category in categories]
    class_palettes = [ 
        (106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
        (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
        (153, 69, 1)
    ]
    
    cfg.metainfo = {
        'classes': tuple(class_names),
        'palette': class_palettes
    }
    
    # 修改训练集路径
    cfg.train_dataloader = dict(
        batch_size=4,
        num_workers=2,
        dataset=dict(
            type='CocoDataset',
            metainfo=cfg.metainfo,
            ann_file=train_data_file,
            data_prefix=dict(img="./data/"),
            pipeline=cfg.train_pipeline
        )
    )

    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            metainfo=cfg.metainfo,
            ann_file=test_data_file,
            data_prefix=dict(img="./data/"),
            test_mode=True,
            pipeline=cfg.test_pipeline
        )
    )

    cfg.val_evaluator = dict(
        type='CocoMetric',
        metric='bbox',
        format_only=False,
        classwise=True,
        ann_file=test_data_file,
        prefix=f'test'
    )

    # 修改预训练模型路径
    cfg.load_from = './checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'

    # 其他训练参数
    cfg.work_dir = f'./work_dirs/faster_rcnn_r101_{source_domain}To{target_domain}_{args.suffix}'
    cfg.optim_wrapper.optimizer.lr = 0.02 / 16
    cfg.train_cfg.max_epochs = 36
    cfg.train_cfg.val_interval = 1
    
    # 修改默认钩子配置
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1),
        sampler_seed=dict(type='DistSamplerSeedHook')
    )
    
    # 更新logger配置
    cfg.default_hooks.update({
        'logger': dict(
            type='LoggerHook',
            interval=100,
            log_metric_by_epoch=True,
        )
    })

    # wandb
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs={
                    'project': 'heart_detection_singleDG',
                    'entity': 'nekok-hkust',
                    'name': f'faster_rcnn_r101_{source_domain}To{target_domain}_{args.suffix}',
                },
                save_dir='wandb'
            )
        ]
    )
    cfg.device = args.device
    
    # 添加调试信息
    print("meta info", cfg.metainfo)
    print("Categories details:", categories)
    print("Number of classes:", num_classes)
    print("Number of training images:", len(annotations['images']))
    print("Number of training annotations:", len(annotations['annotations']))

    # 创建工作目录
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # 创建 Runner 并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
