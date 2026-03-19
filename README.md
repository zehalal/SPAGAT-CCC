# BreastCancer1 运行流程说明

本文档说明 BreastCancer1 数据集从原始数据到最终评估的完整运行流程，共分六个阶段。

---

## 目录结构概览

```
BreastCancer1/
├── 1.Preprocessing/          # 数据预处理
├── 3.LR_Screening/           # LR 基因对筛选
│   ├── 1.Data preprocessing/
│   ├── 3.Build a gene network/
│   └── 4.Identify sensitive genes and gene combinations/
├── 4.LR_Scoring/             # LR 评分
├── 5.Model_Training/         # 模型训练
└── 6.Results/                # 结果评估
    ├── Metric_Evaluation/    # 指标计算
    └── Threshold_Selection/  # 阈值选取
```

---

## Step 1 — 数据预处理

**脚本：** `1.Preprocessing/make_expr_tables.R`

**依赖输入：**
- `10X_bc_Ligs_up_yll.rds` — 各细胞类型的配体基因列表
- `10X_bc_Recs_expr.rds` — 各细胞类型的受体基因列表
- `output_bc.RData` — 包含原始计数矩阵 `de_count`、细胞类型标注 `de_cell_type`、空间坐标 `de_coords`

**主要操作：**
1. 读取细胞类型-基因映射，按每个细胞类型计算配体/受体的平均表达
2. 展开为"基因__细胞类型"为行、所有细胞为列的宽格式矩阵（仅对应细胞类型列填真实值，其余填 0）
3. 导出空间坐标及细胞类型信息

**输出：**
| 文件 | 说明 |
|------|------|
| `de_coords.csv` | 细胞空间坐标 + 细胞类型标注 |
| `ligand_expr_by_cell.csv` | 行=`LigandGene__CellType`，列=所有细胞 |
| `receptor_expr_by_cell.csv` | 行=`ReceptorGene__CellType`，列=所有细胞 |

---

## Step 2 — LR 基因筛选：数据预处理（MATLAB）

### Step 2a — 信号平滑与插值

**脚本：** `3.LR_Screening/1.Data preprocessing/preprocess.m`

**依赖输入：** `ligand_expr_by_cell.csv`（及受体侧同理）

**主要操作：**
1. 读取表达矩阵，滤除低表达基因（均值 ≤ 0.2）及 ERCC 基因
2. 检测不稳定区间并局部重归一化（`p_data`）
3. Log 变换（`10 * log(1 + x)`）
4. 线性插值 5 倍上采样，增加时间/空间分辨率

**输出：** `t_wavelet-all-ori-L100列.mat` / `t_wavelet-all-ori-R100列.mat`（插值后原始特征矩阵）

---

### Step 2b — 小波低频趋势提取

**脚本：** `3.LR_Screening/1.Data preprocessing/preWAVE.m`

**依赖输入：** `t_wavelet-all-ori-L100列.mat` / `t_wavelet-all-ori-R100列.mat`

**主要操作：**
- 对每个基因的表达时间序列进行 5 级 db4 小波分解
- 重建近似系数（低频趋势），去除高频噪声
- 过滤极小值（< 0.0001），取绝对值

**输出：** `t_wavelet-all-L100列.mat` / `t_wavelet-all-R100列.mat`（小波滤波后特征矩阵，包含 `t_data` 和 `point`）

---

## Step 3 — 构建基因调控网络（GCN）

### Step 3a — GCN 训练与最优阈值选取

**脚本：** `3.LR_Screening/3.Build a gene network/GCNtest.py`

**依赖输入：** `t_wavelet-all-L100列.mat`（或 R 侧），包含 `t_data`

**主要操作：**
1. 对候选阈值（0.4 / 0.6 / 0.8），分别用皮尔逊相关构建先验邻接矩阵
2. 训练两层 GCN（1000 epoch × 3 轮），以先验矩阵为监督目标，输出基因对连边概率
3. 从四个维度评估每个阈值：
   - 多轮训练 Jaccard 稳定性
   - 预测网络密度与先验网络密度之差
   - 最大连通分量比例
   - 无孤立节点比例
4. 加权综合评分，取最高分对应的阈值为最优阈值
5. 用最优阈值生成最终二值邻接矩阵

**输出：**
| 文件 | 说明 |
|------|------|
| `threshold_sync_eval-k1-100列.csv` | 各阈值综合评分明细 |
| `L-averaged_final_adj-sync-best-100列.xlsx` | 配体侧最优阈值下的基因邻接矩阵 |
| （R 侧同理）`R-averaged_final_adj-sync-best-100列.xlsx` | 受体侧邻接矩阵 |

---

### Step 3b — 合并邻接矩阵与特征

**脚本：** `3.LR_Screening/3.Build a gene network/getpre.m`

**依赖输入：**
- `t_wavelet-all-L100列.mat`（小波特征）
- `L-averaged_final_adj-k1-100列.xlsx`（GCN 生成的邻接矩阵）

**主要操作：**
- 将小波特征矩阵 `t_data`、邻接矩阵 `maprho`、基因名 `name` 打包保存

**输出：** `prewavelet-k1-L100列.mat` / `prewavelet-k1-R100列.mat`

---

## Step 4 — 识别敏感基因与基因组合（MATLAB）

### Step 4a — 初始网络熵指数计算

**脚本：** `3.LR_Screening/4.Identify sensitive genes and gene combinations/1.Initial network entropy index calculation/c_index0.m`

**依赖输入：** `prewavelet-k1-L100列.mat` / `prewavelet-k1-R100列.mat`

**主要操作：**
1. 以滑动窗口（窗口长度 L=3，训练长度 m=10）遍历时间序列
2. 用 AR 神经网络（ARNN）在每个窗口内预测当前基因的表达，计算预测误差（RMSE）
3. 利用皮尔逊相关找邻居基因，基于误差计算局部网络熵指数 `H`
4. 并行化处理（20 核），加速计算

**输出：** `init_indext_wavelet-L100列.mat`（每个基因的网络熵时序 `H`）

---

### Step 4b — 扰动敏感性分析

**脚本：** `3.LR_Screening/4.Identify sensitive genes and gene combinations/2.Disturbance handling/raodong_point.m`

**依赖输入：**
- `prewavelet-k1-L100列.mat`
- `init_indext_wavelet-L100列.mat`

**主要操作：**
1. 对每个基因施加正负扰动（例如 u = -0.3），再次计算网络熵
2. 对比扰动前后熵的变化量，量化该基因的敏感性
3. 并行化处理（32 核）

**输出：** `adduwaveletk1-100-(all)-L-merged.mat`（含扰动后的熵变量 `H_point_value`）

---

### Step 4c — 子网络探索，生成有效基因组合

**脚本：** `3.LR_Screening/4.Identify sensitive genes and gene combinations/3.Subnetwork exploration/final_test.m`

**依赖输入：**
- `adduwaveletk1-100-(all)-L-merged.mat`（或 R 侧）
- `prewavelet-k1-L100列.mat`

**主要操作：**
1. 按均值熵阈值（< 0.254）筛选低熵（敏感）基因
2. 在筛选后的基因子集上枚举所有子集组合，检测连通性（只保留连通子图）
3. 对各扰动强度 u 计算组合级熵变，筛选出高敏感基因组合

**输出：** `new_name-L100列.mat` / `new_name-R100列.mat`（筛选后的配体/受体基因名列表）

---

### Step 4d — 生成配对候选列表

**脚本：** `3.LR_Screening/4.Identify sensitive genes and gene combinations/3.Subnetwork exploration/make_combo_only.py`

**依赖输入：**
- `new_name-L100列.mat`（筛选后配体基因）
- `new_name-R100列.mat`（筛选后受体基因）

**主要操作：**
- 对配体 × 受体做笛卡尔积组合，去除同细胞类型配对
- 格式化为 `LigGene__SenderCT|RecGene__ReceiverCT`

**输出：** `combo_only 100列.csv`（单列 combo，行数 = 最终候选 LR 对总数）

---

## Step 5 — LR 评分

### Step 5a — 筛选表达矩阵

**脚本：** `4.LR_Scoring/filter_expr_with_matrix筛选ligand_expr_by_cell_filtered.py`

**依赖输入：**
- `combo_only 100列.csv`
- `ligand_expr_by_cell.csv`（来自 Step 1）
- `receptor_expr_by_cell.csv`（来自 Step 1）

**主要操作：**
- 从 combo_only 中提取所有出现的配体/受体特征名，按此过滤原始大矩阵，只保留有效行

**输出：**
| 文件 | 说明 |
|------|------|
| `ligand_expr_by_cell_filtered 100列.csv` | 过滤后的配体表达矩阵 |
| `receptor_expr_by_cell_filtered 100列.csv` | 过滤后的受体表达矩阵 |

---

### Step 5b — GPU 加速的空间 LR 评分

**脚本：** `4.LR_Scoring/run_compute_all_LR_scores_V0_gpu.py`

**依赖输入：**
- `ligand_expr_by_cell_filtered 100列.csv`
- `receptor_expr_by_cell_filtered 100列.csv`
- `de_coords.csv`（细胞空间坐标）

**主要操作（V0 距离加权方案）：**

$$\text{LRscore}_{(L \cdot S,\ R \cdot T)} = \alpha \cdot \sum_i w(d_{ij}) \cdot \text{expr}_L^{(i)} \times \text{expr}_R^{(j)}$$

其中 $w(d)$ 为分段距离权重：
- $d < d_1$（近距离）：固定权重 `w_near`
- $d_1 \le d < d_2$（中距离）：指数衰减 $e^{-\kappa \cdot d}$
- $d \ge d_2$（远距离）：更快衰减 $e^{-\lambda \cdot d}$

使用 CuPy 在 GPU 上计算距离矩阵和评分，支持按 cluster-pair 分块节省显存。

**输出：** `LR_scores_all_pairs_V0_meta_gpu.csv`（所有 LR 配对在每个细胞对上的评分）

---

## Step 6 — 模型训练

**脚本：** `5.Model_Training/run_demo_2layers-LR多头.py`

**依赖输入：**
- `ligand_expr_by_cell_filtered 100列.csv`
- `receptor_expr_by_cell_filtered 100列.csv`
- `LR_scores_all_pairs_V0_meta_gpu.csv`
- `combo_only 100列.csv`

**主要操作：**
1. 将配体/受体过滤矩阵拼接为节点特征矩阵（行=基因节点，列=细胞特征）
2. 利用 kNN 构建对称空间图（基于表达欧氏距离）
3. 用 LR 评分矩阵初始化多头注意力的边权重
4. 调用 STAGATE 两层图注意力网络训练，输出每个 LR 配对的聚合得分

**输出：** `6.Results/Metric_Evaluation/pred_scores.csv`（模型预测的 LR 通讯得分，含 Cell1, Cell2, score 列）

---

## Step 7 — 结果评估

**脚本：** `6.Results/Metric_Evaluation/eval_other_methods.py`

**依赖输入：**
- `pred_scores.csv`（本方法输出）
- `OtherMethods/` 下各基准方法的结果文件（CellChatV2、CytoSignal、COMMOT 等）
- `combo_only 100列.csv`（前 362 行作为已验证正例）

**主要操作：**
1. 将各方法预测得分统一规范化为 `(Cell1, Cell2, score)` 格式
2. 按无向键（`sorted(Cell1, Cell2)` 拼接）与已验证正例做 left join 构建标签
3. 计算全局及 Top-K 指标：
   - ROC-AUC / PR-AUC
   - precision@K / recall@K

**输出：** 控制台打印各方法指标对比表

---

## 快速启动命令

```bash
# Step 1（R）
Rscript BreastCancer1/1.Preprocessing/make_expr_tables.R

# Step 2（MATLAB，顺序执行）
# 在 MATLAB 中 cd 到对应目录后运行
# 3.LR_Screening/1.Data preprocessing/preprocess.m
# 3.LR_Screening/1.Data preprocessing/preWAVE.m

# Step 3（Python）
python "BreastCancer1/3.LR_Screening/3.Build a gene network/GCNtest.py"

# Step 3b（MATLAB）
# 3.LR_Screening/3.Build a gene network/getpre.m

# Step 4（MATLAB + Python）
# c_index0.m → raodong_point.m → final_test.m → make_combo_only.py

# Step 5（Python）
python "BreastCancer1/4.LR_Scoring/filter_expr_with_matrix筛选ligand_expr_by_cell_filtered.py"
python "BreastCancer1/4.LR_Scoring/run_compute_all_LR_scores_V0_gpu.py"

# Step 6（Python，需要 GPU）
python "BreastCancer1/5.Model_Training/run_demo_2layers-LR多头.py"

# Step 7（Python）
python "BreastCancer1/6.Results/Metric_Evaluation/eval_other_methods.py"
```

---

## 数据流向总览

```
output_bc.RData
10X_bc_Ligs_up_yll.rds          Step 1
10X_bc_Recs_expr.rds          ──────────►  ligand_expr_by_cell.csv
                                            receptor_expr_by_cell.csv
                                            de_coords.csv
                                                │
                                           Step 2a/2b（MATLAB）
                                                │
                                    t_wavelet-all-L/R100列.mat
                                                │
                                           Step 3a（GCN）
                                                │
                              L/R-averaged_final_adj-sync-best-100列.xlsx
                                                │
                                           Step 3b（MATLAB）
                                                │
                                     prewavelet-k1-L/R100列.mat
                                                │
                                  Step 4a → 4b → 4c（MATLAB）
                                                │
                                     new_name-L/R100列.mat
                                                │
                                           Step 4d（Python）
                                                │
                                       combo_only 100列.csv
                                                │
                                 ┌──────────────┤
                           Step 5a          Step 5b（GPU）
                                │                │
                  filtered expr CSVs     LR_scores_all_pairs_V0_meta_gpu.csv
                                └──────────────┐
                                           Step 6（STAGATE）
                                                │
                                         pred_scores.csv
                                                │
                                           Step 7（评估）
                                                │
                                    ROC-AUC / PR-AUC / precision@K
```
