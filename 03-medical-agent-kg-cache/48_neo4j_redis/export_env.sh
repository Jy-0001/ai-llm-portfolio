#!/bin/bash

# ① requirements.txt: 只保留普通 pip 包(过滤掉 @file:/// 那种 conda 装的)
pip freeze | grep -v "@ file://" > requirements.txt

# ② requirements_special.txt: 只保留 conda 装的特殊包
pip freeze | grep "@ file://" > requirements_special.txt

# ③ env_info.txt: 系统/CUDA/PyTorch 全景快照
{
    echo "===== Python版本 ====="
    python --version
    echo -e
    
    echo "===== CUDA驱动版本（nvidia-smi）====="
    nvidia-smi | head -3
    echo -e
    
    echo "===== CUDA Toolkit版本 ====="
    nvcc --version
    echo -e
    
    echo "===== PyTorch + CUDA信息 ====="
    python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'PyTorch内置CUDA版本: {torch.version.cuda}')
print(f'CUDNN版本: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU架构(compute capability): {torch.cuda.get_device_capability(0)}')
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
"
    echo -e
    
    echo "===== 操作系统 ====="
    cat /etc/os-release | head -5
    echo -e
    
    echo "===== 关键包版本 ====="
    pip show torch       | head -2
    pip show transformers | head -2
    pip show deepspeed   | head -2
    echo -e
    
    echo "===== 完整pip freeze ====="
    pip freeze
} > env_info.txt

echo "✅ 生成完成:"
ls -la requirements.txt requirements_special.txt env_info.txt
