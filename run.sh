#!/bin/bash
# 自动遍历 tau 参数值的 Shell 脚本（无引号版）

tau_values=(5 10 20 100)

for tau in ${tau_values[@]}; do
    echo "---------------------------------------------"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 启动任务: tau = $tau"
    
    # 执行命令（所有参数均不加引号）
    python /root/jsc/SCRNN/mysnn.py \
        --network SimpleMemoryNetwork_2024 \
        --layer_A \
        --mode analyse \
        --tau $tau
    
    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo "[成功] tau=$tau 任务已完成"
    else
        echo "[失败] tau=$tau 任务执行出错！" >&2
    fi
done

echo "---------------------------------------------"
echo "所有 tau 参数任务执行完毕！"