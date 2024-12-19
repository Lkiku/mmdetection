#!/bin/bash

# 获取当前用户名
current_user=$(whoami)

# 获取所有在GPU上运行的进程ID
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

if [ -z "$gpu_pids" ]; then
    echo "没有发现GPU上运行的进程"
    exit 0
fi

echo "正在检查属于用户 $current_user 的GPU进程:"
for pid in $gpu_pids; do
    # 检查进程是否属于当前用户
    pid_user=$(ps -o user= -p $pid)
    if [ "$pid_user" = "$current_user" ]; then
        process_name=$(ps -p $pid -o comm=)
        echo "终止进程 - PID: $pid ($process_name)"
        kill -9 $pid
    fi
done

echo "所有属于用户 $current_user 的GPU进程已终止"
