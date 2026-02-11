
# 图片整理器 v2.0（CLI + GUI）

根据图片内容自动分类到：

# Image Organizer

一个按图片内容自动分类的小工具，会将图片整理到：


- `people`（人物）
- `scenery`（风景）
- `objects`（物品）
- `unknown`（低置信度）


## 2.0 更新

- 新增 **桌面 GUI**：三步完成（选源目录 → 选输出目录 → 点开始）。
- 保留 CLI，适合批处理和脚本化。
- 新增 `run.py` 一键启动，会先检查并自动安装依赖，再打开 GUI。

## 最简部署（推荐）

```bash
python run.py
```

> 首次运行会下载模型权重，耗时取决于网络环境。

## 手动安装

## 安装


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## GUI 模式

```bash
python organize_images.py --gui
```

## CLI 模式

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

- `--min-confidence 0.2`：最低置信度（0~1），低于阈值放入 `unknown`

## 分类原理

- 使用 `torchvision` 预训练 `MobileNetV3-Small`。
- 对每张图片做 ImageNet 标签预测。
- 通过关键词映射到人物/风景/物品类别；低置信度归入 `unknown`。

- `--min-confidence 0.2`：最低置信度，低于阈值放到 `unknown`

## 说明

- 模型使用 `torchvision` 预训练 `MobileNetV3-Small`。
- 分类逻辑：先识别 ImageNet 标签，再映射为人物/风景/物品三类。
- 首次运行可能下载模型权重。

