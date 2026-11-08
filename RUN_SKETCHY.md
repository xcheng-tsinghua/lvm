# Sketchy 本地运行

## 根目录 train.py（SwanLab）

```powershell
conda activate dp
# 类别级任务：新建实验，从头训练
python train.py --dataset sketchy --task cl --epoch 60 --batch_size 16 --seed 42 --swanlab_api_key YOUR_API_KEY --swanlab_project Sketch_LVM --resume False
# 细粒度任务
python train.py --dataset sketchy --task fg --epoch 60 --batch_size 16 --seed 42 --swanlab_api_key YOUR_API_KEY --swanlab_project Sketch_LVM --resume False
# 恢复最近匹配的类别级实验，继续至总共 60 轮
python train.py --dataset sketchy --task cl --epoch 60 --batch_size 16 --seed 42 --swanlab_project Sketch_LVM --resume True
```

API Key 也可通过 `SWANLAB_API_KEY` 环境变量或已有 SwanLab 登录提供；恢复命令同样需要认证。脚本不会把 API Key 写入训练配置或 checkpoint。

- `--dataset` 当前支持 `sketchy`；`--data_dir` 默认使用本文的本地数据目录。
- `--task` 支持 `cl` / `CL-ZS-SBIR`、`fg` / `FG-ZS-SBIR`，默认 `cl`。
- `--epoch` 是目标**总轮数**；`--resume False` 新建目录和实验，不覆盖已有训练。`--resume True`（或只写 `--resume`）自动搜索 `runs/train` 中同数据路径、任务、种子、项目及记录模式的最近 checkpoint；没有匹配项会报错。
- 恢复 adapter、Adam、GradScaler、已完成轮数、Python/NumPy/PyTorch/CUDA 随机状态，并以保存的 run ID 和 `resume='must'` 续写同一个 SwanLab 实验。batch size、每类样本数、采样预算、评测限制、workers 与基础权重需保持一致。`--run_dir` 可指定另一组实验目录。
- 每轮结束进行检索评测，以 epoch 为 SwanLab step，上传训练损失以及 CL 的 **`mAP@200`、`precision@200`**，或 FG 的 **`Acc@1`、`Acc@5`**。四个指标均使用 **0–1** 范围（乘 100 即百分比）。
- checkpoint 在每轮评测完成后原子保存；中途退出会从最近完成的轮次恢复，当前未完成轮次需重跑。checkpoint 内保留待写入的指标，日志调用失败后可恢复补写。
- 旧入口生成的 checkpoint 没有 SwanLab run ID，不会被此入口自动认作可续写实验。
- 默认在线记录；`--swanlab_mode offline` 用于无网络调试，离线与在线实验分开匹配。`--steps_per_epoch 1 --eval_limit 1` 可做快速验证，正式训练不要加这两个限制。

验证：6 项单元测试通过；实际 GPU 运行验证了 CL 新建实验、恢复到第 2 轮以及 FG 指标写入，使用 SwanLab 0.10.2 离线记录。云端上传未使用真实 API Key 验证。恢复接口依据 [SwanLab 官方文档](https://docs.swanlab.cn/api/py-init.html)。

新增入口 `python -m experiments.sketchy` 使用原仓库的 CLIP ViT-B/32 和浅层视觉提示，在原生 PyTorch 中运行，避免旧 Lightning 1.x 接口与当前环境冲突。原入口未替换。

## 环境与数据

- Python: `D:\miniconda\envs\dp\python.exe`；PyTorch 2.4.1+cu121；RTX 4060 Ti 16 GB。
- 新增依赖：ftfy 6.3.1、regex 2026.11.7。
- CLIP 基础权重默认从 `~/.cache/clip/ViT-B-32.pt` 加载并验证官方 SHA256；可用 `--weights` 指定本地权重。
- 数据根目录：`D:\document\DataSet\sketch_retrieval\sketchy`，自动识别 `sketch_png` 或 `sketch`，照片使用 `photo`。
- 本地共 125 类、12,500 张照片、75,481 张草图。逐条验证草图文件名去掉末尾 `-数字` 后存在同类配对照片；无缺失。
- 使用原仓库的固定 104/21 类划分，训练类别与测试类别不相交。
- 本地数据是 basic Sketchy，**不含论文类别级实验的额外 60,502 张照片**；本地 CL 指标不能直接对比 Sketchy-extended 的论文表格。

## 已运行的端到端短训

```powershell
conda activate dp
python -m experiments.sketchy --task cl --output runs/cl_sketchy --steps-per-epoch 100 --batch-size 16
python -m experiments.sketchy --task fg --output runs/fg_sketchy --steps-per-epoch 100 --batch-size 16
```

每个任务进行 100 次优化更新，并在全部 21 个未见类别上评测；这属于跑通验证，不是收敛训练，也不宣称复现论文精度。训练不使用测试集选择 checkpoint。

实际结果（2026-11-08；全部 12,694 条测试查询、2,100 张唯一照片）：

| 任务 | 指标 | 结果 |
|---|---|---|
| CL-ZS-SBIR / basic Sketchy | mAP@all | 0.321258 |
| CL-ZS-SBIR / basic Sketchy | mAP@200 | 0.260470 |
| CL-ZS-SBIR / basic Sketchy | P@200 | 0.227687 |
| FG-ZS-SBIR / 类内检索 | Top-1 | 1.9931% |
| FG-ZS-SBIR / 类内检索 | Top-5 | 7.8935% |

CL 和 FG 的训练阶段分别耗时约 125.6 秒和 122.2 秒（不含特征提取与评测）。FG 的宏平均 Top-1/Top-5 为 1.9822% / 7.8559%。短训精度较低，不用于声称达到论文效果。

输出目录包含：`manifest.json`（配置、划分及环境），`train.jsonl`（各项损失），`last.pt`（可训练参数与优化器状态），`metrics.json`（评测及 checkpoint 哈希），`features.pt`（固定图库、查询路径与特征）。权重文件仅保存 adapter，加载时仍需要相同的 CLIP 基础权重。

## 独立评测与继续训练

```powershell
python -m experiments.sketchy --task cl --output runs/cl_eval --eval-only --checkpoint runs/cl_sketchy/last.pt
python -m experiments.sketchy --task fg --output runs/fg_eval --eval-only --checkpoint runs/fg_sketchy/last.pt
```

较长训练示例（耗时明显增加）：

```powershell
python -m experiments.sketchy --task cl --output runs/cl_60epochs --epochs 60 --batch-size 64
python -m experiments.sketchy --task fg --output runs/fg_60epochs --epochs 60 --batch-size 32
```

`--steps-per-epoch 0` 为按训练集大小计算的一轮随机平衡采样预算；不是逐样本无放回遍历。默认每类 4 个样本，batch size 须是 4 的倍数且至少为 8。FG 使用六路前向，显存不足时减小 batch。`--workers 0` 为 Windows 兼容默认值；可增大以提高读取速度。

继续训练加 `--checkpoint runs/fg_sketchy/last.pt --epochs 60`。`--epochs` 为总轮数而非新增轮数。短训 checkpoint 的第一轮只有 100 步，继续它不等价于从头进行 60 个完整预算轮次。

## 方法与协议

- CL：独立 sketch/photo 提示及独立视觉 LayerNorm，类别级 triplet 和冻结 CLIP 文本分类损失；除提示和视觉 LayerNorm 外全部冻结。
- FG：共享提示与视觉 LayerNorm，配对照片作为正例、同类别不同照片作为负例；文本分类、相对距离 KL 及 2×2 patch shuffle triplet。
- margin 0.3，prompt lr 1e-5，LN lr 1e-6，分类权重 0.5，KL 权重 0.1，patch loss 权重 1。单个文本模板为 `a photo of a {category}`。使用原代码 ImageNet 归一化和 ImageOps.pad，未改为原生 CLIP center crop。
- KL 采用类别平衡 batch，将各类等长相对距离向量 softmax，再计算所有有序类别对 KL 的平均值。论文未提供完整采样细节；这是明确的实现选择。
- 每个样本随机生成一个 patch 排列，另一个排列由其循环移位得到，保证正负排列不同。
- CL：21 类全部唯一照片构成统一图库。报告 mAP@all、mAP@200、P@200；AP@200 分母明确定义为 `min(200, relevant_gallery_count)`。
- FG：每类单独在其 100 张照片中检索配对实例；报告按全部查询加权的 Top-1/Top-5，同时输出类平均值和逐类统计。
- 排序采用归一化特征的余弦相似度，并稳定处理并列结果。

**来源与限制**：仓库 README 明确说只发布 workable basic version，FG 代码待发布。新增分类损失、分支 LayerNorm 和 FG 训练是依据[论文第 4–6 节](https://arxiv.org/html/2303.13440v1)的本地实现，不是作者未发布源码。论文使用 batch size 64、60 epochs；这里短训使用 batch size 16、100 updates。数据中的草图数与论文给出的 75,471 相差 10，本实现保留本地全部文件并检查配对，不擅自删样本。

## 验证

```powershell
python -m unittest experiments.test_sketchy
```

测试覆盖 patch shuffle 可逆性、triplet 正负方向、KL 恒等分布及非平凡梯度。GPU 冒烟运行还覆盖完整反向传播、adapter 保存和真实图片检索。

四个单元测试全部通过，其中固定特征样例额外验证了类别级 AP 与类内实例排名。两个训练 checkpoint 均已重新加载，并与原评测保存的查询特征核对：最大绝对误差分别为 0.000342 和 0.000335（不同 batch 大小下的混合精度误差）。详见 `runs/checkpoint_verification.json`。CL/FG 可训练参数数目分别为 84,480 / 42,240，保存参数全部有限。
`runs/fg_resume_check` 还完成了从 FG checkpoint 恢复优化器、GradScaler 与随机状态后的 1 步续训及评测，退出码为 0；该目录仅用于续训验证，不替代主结果。
