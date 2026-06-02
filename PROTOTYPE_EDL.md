# Prototype + EDL

本文档说明当前仓库里的 Prototype + Evidential Deep Learning 路径，以及做消融时建议关注的参数。

当前实际入口：

- 训练主脚本：`finetune_proto_edl.py`
- 启动脚本：`run_proto_edl.py`
- 模型头：`prototype_edl_model.py`

## 核心流程

Prototype + EDL 保留 Mammo-FM backbone，只替换分类头。

```text
image
  -> embedding z
  -> class-wise prototype distance
  -> prototype similarity
  -> prototype evidence
  -> class evidence
  -> alpha / probability / uncertainty
```

主判断列：

- `pred_score = probability[:, 1]`
- `pred_label = pred_score >= 0.5`

当前实现里，Prototype EDL 的 validation 聚合、最终 test metric 和 prediction 口径都已尽量对齐 `origin` / `EDL` 路径。

## Prototype 正则

训练时的 prototype 正则包含三项：

1. `attract`：样本靠近同类最近 prototype
2. `separation`：样本远离异类最近 prototype
3. `diversity`：同类 prototype 之间保持分散

分项原始组合为：

```text
prototype_loss_raw =
    edl_proto_attract_weight * attract
  + edl_proto_separation_weight * separation
  + edl_proto_diversity_weight * diversity
```

实际进入总 loss 的 prototype 项为：

```text
prototype_loss = edl_proto_loss_weight * prototype_loss_raw
total_loss = edl_loss + class_loss + prototype_loss
```

说明：

- `proto_raw` 表示未乘总权重的 prototype regularization
- `proto` 表示乘完 `edl_proto_loss_weight` 之后，真正进入总 loss 的 prototype loss

## 默认参数

当前默认值：

```text
edl_proto_k = 4
edl_proto_topk = 3
edl_proto_temperature = 1.0
edl_proto_normalize = y

edl_proto_class_weight = 1.0
edl_proto_attract_weight = 0.1
edl_proto_separation_weight = 0.1
edl_proto_diversity_weight = 0.01
edl_proto_loss_weight = 1.0
edl_proto_margin = 1.0
edl_proto_balance_classes = y
```

其中：

- `edl_proto_loss_weight=1.0` 表示保持原始 prototype 正则比例
- 若要放大 prototype 模块整体影响，优先调 `edl_proto_loss_weight`

## 推荐调参顺序

做消融时建议按下面顺序来：

1. 先固定分项默认值 `0.1 / 0.1 / 0.01`
2. 先试 `--edl-proto-loss-weight 3.0`
3. 如果 `proto_raw` 本身就很小，再调分项权重
4. 如果 `proto_raw` 合理，但 `proto` 相对 `edl` 还是太弱，再试 `--edl-proto-loss-weight 5.0`

一个实用判断标准是：

- 先看 `train_proto_reg_loss_raw`
- 再看 weighted 的 `train_proto_reg_loss`
- 观察 weighted `proto` 相对 `edl` 是否进入一个可见区间

## 日志与曲线

当前训练 history / TensorBoard 会记录：

- `train_total_loss`
- `train_edl_loss`
- `train_class_loss`
- `train_proto_reg_loss`
- `train_proto_reg_loss_raw`
- `train_proto_attract_loss`
- `train_proto_separation_loss`
- `train_proto_diversity_loss`
- `edl_proto_loss_weight`

其中：

- `train_proto_reg_loss` 是 weighted prototype loss
- `train_proto_reg_loss_raw` 是未乘总权重的 raw prototype loss

component loss curve 会同时画出 raw 和 weighted，方便看“原始量级太小”还是“总权重太小”。

## 命令示例

直接跑启动脚本：

```bash
python run_proto_edl.py
```

如果想放大 prototype 模块总权重：

```bash
python run_proto_edl.py --edl-proto-loss-weight 3.0
```

或者直接调用训练脚本：

```bash
python finetune_proto_edl.py \
  --data-dir /path/to/data \
  --csv-file data.csv \
  --edl-proto-k 4 \
  --edl-proto-loss-weight 3.0
```

## 输出目录

默认 `edl_proto_loss_weight=1.0` 时，不改现有输出目录。

只有当总权重不是 `1.0` 时，输出目录会自动追加后缀，例如：

```text
output_proto_w3.0
best_model_proto_w3.0
```

这样默认实验和放大权重实验可以自然分开，不会覆盖旧结果。
