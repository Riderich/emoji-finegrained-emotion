@echo off
REM 数据准备脚本：下载 OpenMoji 面部 Emoji 并进行数据增强
REM 说明：需要已配置好 conda 环境 cuda_env

IF "%CONDA_DEFAULT_ENV%"=="" (
  echo [提示] 建议激活或使用 conda 运行指定环境：cuda_env
)

REM 下载 OpenMoji 面部 Emoji 到 data/emoji_images/openmoji/
conda run -n cuda_env python -m src.data.download_openmoji || goto :error

REM 对下载的图像进行旋转/亮度/对比度增强输出到 data/emoji_images/openmoji_aug/
conda run -n cuda_env python -m src.data.augment_images --src data/emoji_images/openmoji --dst data/emoji_images/openmoji_aug || goto :error

echo [完成] OpenMoji 下载与增强已完成。
exit /b 0

:error
echo [错误] 脚本执行失败，请检查依赖与网络连接。
exit /b 1