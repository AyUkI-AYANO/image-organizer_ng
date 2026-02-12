# 图片整理器 NG 5.1.0（CLI + Web UI）

一个按图片内容自动分类的小工具，会将图片整理到：

- `people`（人物）
- `scenery`（风景）
- `animals`（动物）
- `vehicles`（交通工具）
- `food`（食物）
- `buildings`（建筑）
- `objects`（其他物品）
- `unknown`（低置信度或低匹配分）

## NG 5.1.0 重点升级

- **算法优化**：分类由简单 Top-1 判断升级为 **Top-K 融合 + 分层语义加权**，并引入同义词归一化，降低标签噪声导致的误分类。
- **精度优化**：
  - Rank 衰减：高排名预测更高权重，避免低概率标签干扰。
  - 置信度平滑：使用平方根平滑，抑制极端置信值对结果的“过拟合”。
  - 词级重叠 + 短语精确匹配 + 子串弱匹配组合打分，更稳健。
- **UI 全量升级**：从 Tk GUI 升级到 **现代化 Gradio Web UI**，交互更直观、参数调节更方便。
- **新增 Dry-Run 模式**：可先预演分类与输出路径，不写入文件，便于安全验证。
- **新增 JSON 报告导出**：可输出完整处理记录（含每张图的预测标签、置信度、目标路径和执行动作），方便审计与复盘。

## 最简部署（推荐）

```bash
python run.py
```

> 首次运行会下载模型权重，耗时取决于网络环境。

## 手动安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web UI 模式

```bash
python organize_images.py --gui
```

启动后打开浏览器访问：

- `http://127.0.0.1:7860`

## CLI 模式

```bash
python organize_images.py <图片目录> -o <输出目录>
```

示例：

```bash
python organize_images.py ./photos -o ./sorted_photos --min-confidence 0.22 --min-category-score 0.16 --topk 10

# 仅预演，不复制/移动文件，并导出报告
python organize_images.py ./photos -o ./sorted_photos --dry-run --report ./reports/run_report.json
```

可选参数：

- `--move`：移动文件（默认复制）
- `--min-confidence 0.22`：模型 Top-1 最低置信度（0~1），低于阈值放入 `unknown`
- `--min-category-score 0.16`：类别语义匹配最低分（0~1），低于阈值放入 `unknown`
- `--topk 10`：参与融合的 Top-K 预测数量（建议 8~12）
- `--dry-run`：仅预演分类，不执行复制/移动
- `--report ./reports/run_report.json`：导出 JSON 报告（包含处理明细与统计）

## 报告示例（JSON）

`--report` 生成的 JSON 包含：

- 运行配置：阈值、Top-K、源目录、输出目录
- 汇总信息：总处理数量、类别分布
- 明细清单：每张图的 `source` / `destination` / `category` / `label` / `confidence` / `category_score` / `action`

## 分类原理（NG 5.1.0）

- 使用 `torchvision` 预训练 `MobileNetV3-Small`。
- 对每张图片提取 Top-K ImageNet 标签及置信度。
- 对标签做标准化与同义词归一化（如 `automobile -> car`）。
- 对每个类别进行三层评分：
  - 精确短语匹配（高权重）
  - 词级重叠匹配（中权重）
  - 子串弱匹配（补偿权重）
- 每个标签分数融合 `rank_weight` 和 `confidence_weight` 后累计。
- 同时满足 `min-confidence` 与 `min-category-score` 才进入目标类别，否则归入 `unknown`。
