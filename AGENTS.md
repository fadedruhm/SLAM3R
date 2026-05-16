# SLAM3R — 给 AI 代理的快速指南（中文）

## 目标
为协助开发与实验的 AI 代理提供最小、可操作的导航信息：告诉代理从哪里开始查看、常用运行命令、以及不应重复或修改的内容。

## 先看哪里（优先级）
- [app.py](app.py)：Gradio 演示与在线/离线入口。
- [recon.py](recon.py)：离线/在线重建的独立脚本，常用于实验流水线。
- [train.py](train.py)：训练入口，包含训练参数说明。
- [slam3r/](slam3r)：核心模型、推理与管线实现。
- 文档：参见 [README.md](README.md)、[docs/data_preprocess.md](docs/data_preprocess.md)、[docs/recon_tips.md](docs/recon_tips.md)

## 常用命令（开发/实验）
- 克隆并安装依赖：

  conda create -n slam3r python=3.11
  conda activate slam3r
  pip install -r requirements.txt

- 运行离线重建示例（参考 README 与 scripts 中的 wrapper）：

  python recon.py --img_dir path/to/images --test_name myrun

- 启动演示界面：

  python app.py           # 离线模式
  python app.py --online  # 在线模式

## 开发与编辑原则
- 不要在代理说明中重复 README 或大量文档内容；引用它们即可。
- 修改代码时：尽量将改动局限在单一模块；如果需要更改公共接口（如脚本参数），请同时更新对应的 shell wrapper（scripts/）和文档。
- 不要把中间产物（results/, tmp/, output/）当作持久配置或输入。

## 环境与常见问题提示
- GPU/CUDA 相关：项目使用 `pycuda`，部分可选加速依赖（如 xformers）会影响运行结果；在报告问题时先确认 CUDA、PyTorch 与依赖版本是否匹配。
- 自定义 CUDA 内核：`slam3r/pos_embed/curope/` 下有编译扩展，编译失败请查看 README 中的建议和对应 issue 链接。
- 模型权重：默认会从 HuggingFace 自动下载预训练权重；也支持通过命令行传入本地权重路径。

## 小心事项
- 改动模型或推理函数签名会影响 `recon.py`、`app.py`、以及 `slam3r/pipeline/` 中的多个模块，请同时运行示例来验证。
- 数据预处理脚本分散在 `datasets_preprocess/`，若修改格式或添加选项，请更新相关说明文档。

## 我该怎么继续（建议）
1. 如果你希望我用中文进一步补充每个脚本的快捷运行示例，我可以在 `docs/` 下添加小节（需你确认）。
2. 需要将本文件移动到 `.github/copilot-instructions.md`？（通常 AGENTS.md 已足够）

## Agent 使用说明（汇总）

下列为从仓库 README/Docs 汇总出的快捷指引，供 AI 代理或接手者快速上手：

**快速可执行命令**
```bash
# 环境
conda create -n slam3r python=3.11
conda activate slam3r
pip install -r requirements.txt

# 运行 demo（Replica 示例）
bash scripts/demo_replica.sh

# 启动 Gradio 离线界面
python app.py

# 启动 Cesium 本地服务
python cesium/server.py
```

**代理/开发建议（简明）**
- 优先引用并复用 `README.md` 与本文件 `AGENTS.md`，避免重复粘贴大量文档内容。
- 修改代码时：避免变更模型/推理函数签名（会影响 `recon.py`、`app.py`、`slam3r/pipeline/`）。
- 若需编译扩展，请参见 `slam3r/pos_embed/curope/` 的说明；常见问题先核验 CUDA 与 PyTorch 版本。
- 不要将 `results/`, `tmp/`, `output/` 等中间产物当作持久输入来源。

**常见问题速查**
- GPU/CUDA 与 PyTorch/xformers 版本不兼容会导致运行或精度差异，先确认驱动与环境。
- Cesium 可视化可能出现坐标轴/朝向错位，参阅 `trans-gps/阶段总结.md` 中的 X/Z 轴补偿经验。

（此节为仓库内自动生成汇总，最后修改：2026-05-15）
