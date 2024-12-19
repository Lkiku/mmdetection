#!/bin/bash

# 定义所有域
DOMAINS=("voch1" "voch2" "voch3")
GPU_IDS=(1 2 3)  # 使用三张GPU卡
TRAIN_SCRIPT="train_heart.py"
SUFFIX="baseline"

# 创建logs目录（如果不存在）
mkdir -p logs

# 计数器，用于在GPU间分配任务
counter=0

# 对每个域作为源域进行训练
for SOURCE in "${DOMAINS[@]}"; do
    # 对其他域作为目标域进行测试
    for TARGET in "${DOMAINS[@]}"; do
        # 跳过源域等于目标域的情况
        if [ "$SOURCE" != "$TARGET" ]; then
            # 计算使用哪张GPU（循环使用GPU_IDS中的GPU）
            GPU_INDEX=$((counter % ${#GPU_IDS[@]}))
            GPU_ID=${GPU_IDS[$GPU_INDEX]}
            
            LOG_FILE="logs/${SOURCE}To${TARGET}_${SUFFIX}.log"
            
            echo "Starting experiment: ${SOURCE} to ${TARGET} on GPU ${GPU_ID}"
            echo "Log file: ${LOG_FILE}"
            
            # 使用nohup在后台运行实验，使用CUDA_VISIBLE_DEVICES限制GPU可见性
            nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python ${TRAIN_SCRIPT} \
                --source ${SOURCE} \
                --target ${TARGET} \
                --suffix ${SUFFIX} \
                --device cuda:0 > "${LOG_FILE}" 2>&1 &
            
            # 更新计数器
            counter=$((counter + 1))
            
            # 稍微延迟以避免同时启动可能造成的资源竞争
            sleep 2
        fi
    done
done

echo "All experiments started in background!"
echo "To monitor the experiments, use one of:"
for SOURCE in "${DOMAINS[@]}"; do
    for TARGET in "${DOMAINS[@]}"; do
        if [ "$SOURCE" != "$TARGET" ]; then
            echo "  tail -f logs/${SOURCE}To${TARGET}_${SUFFIX}.log"
        fi
    done
done
