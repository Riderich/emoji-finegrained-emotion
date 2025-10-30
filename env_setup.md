# Windows 环境安装指南（Python 项目）

## 前置条件
- 已安装 Python 3.10 或以上（建议 64-bit）
- 可选：已安装 Git、PowerShell 5+（系统默认）

## 步骤 1：创建虚拟环境
```
python -m venv .venv
```
激活环境：
```
.\.venv\Scripts\activate
```
停用环境：
```
deactivate
```

## 步骤 2：安装依赖
```
pip install --upgrade pip
pip install -r requirements.txt
```
如遇到 `torch` 安装缓慢或失败，可先安装纯 CPU 版本或参考官方轮子：
- 官方帮助：https://pytorch.org/get-started/locally/

## 步骤 3：验证安装
```
python -c "import torch, transformers, cv2; print('OK', torch.__version__)"
```
看到 `OK` 输出即表示核心依赖加载成功。

## 常见问题
- 若 `opencv-python` 安装失败，先升级 `pip` 与 `setuptools` 再重试。
- 若中文路径导致编码问题，建议将数据路径配置在英文目录名下或使用原始路径的短路径格式。

## 开发与运行
- 训练：`scripts\run_train.bat`
- 评估：`scripts\run_eval.bat`

如需 GPU：请确保已安装匹配版本的 CUDA 与 cuDNN，并按 PyTorch 官方指引安装对应版本的 `torch`。