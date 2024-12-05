#!/bin/bash
# 本脚本必须在user用户下运行
# 本脚本实现以下功能
# 1.自动配置16G的swap内存（若已配置名为swapfile的文件则跳过）
# 2.自动创建名为mindspore的conda环境，并配置环境变量在打开终端后自动进入该环境
# 3.自动配置control CPU的个数为4，AI CPU的数量为0
# 4.自动配置静态IP为192.168.137.100,子网掩码为255.255.255.0
SWAPFILE="/swapfile"

# 检查swap文件是否存在，若不存在，则直接创建
if [ -e "$SWAPFILE" ]; then
    echo "Swap file already exists. Skipping swap configuration."
else
    # 创建并配置swap文件
    sudo fallocate -l 16G "$SWAPFILE"
    if [ $? -ne 0 ]; then
        echo "Failed to create swap file."
        exit 1
    fi

    sudo mkswap "$SWAPFILE"
    if [ $? -ne 0 ]; then
        echo "Failed to create swap area on swap file."
        exit 1
    fi

    sudo swapon "$SWAPFILE"
    if [ $? -ne 0 ]; then
        echo "Failed to enable swap file."
        exit 1
    fi

    sudo chmod 600 "$SWAPFILE"
    if [ $? -ne 0 ]; then
        echo "Failed to set permissions on swap file."
        exit 1
    fi

    echo "'$SWAPFILE none swap sw 0 0' " | sudo tee -a /etc/fstab
    if [ $? -ne 0 ]; then
        echo "Failed to add swap file to /etc/fstab."
        exit 1
    fi
fi

if ! conda env list | grep -q mindspore; then
    # 如果不存在，则创建名为mindspore环境，并设置重启shell后自启动该环境
    conda create -n mindspore python=3.9 -y
    echo "conda activate mindspore" >> ~/.bashrc
    echo "mindspore环境已创建。"
else
    echo "mindspore环境已存在。"
fi

# 配置control CPU的数量为4
sudo npu-smi set -t cpu-num-cfg -i 0 -c 0 -v 0:4:0

# 配置ip
sudo nmcli con mod "Wired connection 1" \
ipv4.addresses "192.168.137.100/24" \
ipv4.gateway "192.168.137.1" \
ipv4.dns "8.8.8.8" ipv4.method "manual"

sudo reboot