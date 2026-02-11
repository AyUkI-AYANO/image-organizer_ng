# Image Organizer

一个按图片内容自动分类的小工具，会将图片整理到：

- `people`（人物）
- `scenery`（风景）
- `objects`（物品）
- `unknown`（低置信度）

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使用

```bash
python organize_images.py <图片目录> -o <输出目录>
```

示例：

```bash
python organize_images.py ./photos -o ./sorted_photos
```

可选参数：

- `--move`：移动文件（默认复制）
- `--min-confidence 0.2`：最低置信度，低于阈值放到 `unknown`

## 说明

- 模型使用 `torchvision` 预训练 `MobileNetV3-Small`。
- 分类逻辑：先识别 ImageNet 标签，再映射为人物/风景/物品三类。
- 首次运行可能下载模型权重。
