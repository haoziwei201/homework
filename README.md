# 实验一：文本分类

## 文件运行结构

```
├── svm_experiment.py       # SVM 实验全流程（独立运行，输出 output_svm/，内存有SVM模型预测结果，绘制图表等相关内容）
├── lr_experiment.py        # 逻辑回归实验全流程（独立运行，输出 output_lr/，内存有LR模型预测结果、绘制图表等相关内容）；
│                           #   LR 为报告选定的提交模型，同时会在本目录生成最终predictions.csv）
├── train_data.csv          # 训练数据 7368 条（10 类均衡）
├── test_data_unlabeled.csv # 无标签测试数据 2457 条
├── predictions.csv         # 最终提交的预测（由 lr_experiment.py 生成覆盖示例）
├── output_svm/             # SVM 实验输出（figures / tables / predictions.csv / summary.json / run_log.txt）
└── output_lr/              # LR 实验输出（同上）
```

## 环境与运行

- Python 版本 3.11.9
- 两个实验脚本相互独立，可按任意顺序单独运行（在本目录下）：

```bash
python svm_experiment.py   
python lr_experiment.py   
```

- **断点续跑**：每个脚本的评估结果实时写入各自 `output_*/tables/checkpoint_rows.jsonl`，中途中断后重新运行会自动跳过已完成的训练，只补齐缺失部分。

LR C=10 验证集 0.9383/0.9384 略优于 SVM linear C=10 的 0.9369/0.9372，且训练快约一个数量级、可输出概率——综合验证性能与模型复杂度选定 LR 为提交模型。

## 输出图表

| 位置 | 内容 |
| output_svm/figures/ | figS1 清洗消融 · figS2 TF-IDF 网格 · figS3 C 曲线 · figS4 混淆矩阵 · figS5 hinge 损失曲线 |
| output_lr/figures/  | figL1 清洗消融 · figL2 TF-IDF 网格 · figL3 C 曲线 · figL4 混淆矩阵 · figL5 交叉熵损失曲线 |

## 可复现性说明

- 全局 `SEED=42`作为随机种子；两个脚本使用完全相同的划分与特征配置，结果可逐项复现；
- 每次评估的训练/预测耗时均有记录（`output_*/tables/all_results.csv`）；
- 删除 `output_svm/`、`output_lr/` 后重新运行对应脚本，即可从干净状态复现全部表格与图表；
