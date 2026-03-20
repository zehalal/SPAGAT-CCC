import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


def load_known_pairs(path: str, nrows: int = 362) -> pd.DataFrame:
	"""读取已验证正例（前 nrows 行）。支持：
	- 单列 'Ligand|Receptor'
	- 多列时，取第1、2列作为 Cell1/Cell2
	- 自动跳过不含 '|' 的行
	"""

	raw = pd.read_csv(path, nrows=nrows, header=None)

	if raw.shape[1] == 1:
		col = raw.iloc[:, 0].astype(str)
		mask = col.str.contains("|")
		split = col[mask].str.split("|", n=1, expand=True)
	else:
		# 取前两列为 Cell1/Cell2
		split = raw.iloc[:, :2].copy()
		split.columns = [0, 1]

	if split.shape[1] != 2:
		raise ValueError("无法拆分出 Cell1/Cell2，请检查 combo_only 文件格式")

	known = split.rename(columns={0: "Cell1", 1: "Cell2"})
	known = known.dropna(subset=["Cell1", "Cell2"])
	known["Cell1"] = known["Cell1"].astype(str).str.strip()
	known["Cell2"] = known["Cell2"].astype(str).str.strip()
	known["label"] = 1
	known["key_dir"] = known["Cell1"] + "||" + known["Cell2"]
	known["key_ud"] = known.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
	return known


def load_pred_scores(path: str) -> pd.DataFrame:
	"""读取模型预测文件，要求列名含 Cell1/Cell2 和分数字段（att 或 score）。"""

	pred = pd.read_csv(path)
	score_col = "score"
	if score_col not in pred.columns:
		if "att" in pred.columns:
			score_col = "att"
		else:
			raise ValueError("pred_scores.csv 需包含 score 或 att 列")

	pred = pred.rename(columns={score_col: "score"})
	pred["Cell1"] = pred["Cell1"].astype(str).str.strip()
	pred["Cell2"] = pred["Cell2"].astype(str).str.strip()
	pred["key_dir"] = pred["Cell1"] + "||" + pred["Cell2"]
	pred["key_ud"] = pred.apply(lambda r: "||".join(sorted([r.Cell1, r.Cell2])), axis=1)
	return pred


def main():
	# 使用脚本所在目录，避免工作目录不同导致找不到文件
	pkg_dir = Path(__file__).resolve().parent
	known = load_known_pairs(pkg_dir.parent / "5.Model_Training" / "combo_only 100列.csv", nrows=362)
	pred = load_pred_scores(pkg_dir / "pred_scores.csv")

	# 构造全集标签：全集 = pred 里的所有 key
	# 先用无序 key_ud 匹配，避免方向不一致导致漏匹配
	labels = pd.DataFrame({"key_ud": pred["key_ud"].unique()})
	labels = labels.merge(known[["key_ud", "label"]], on="key_ud", how="left")
	labels["label"] = labels["label"].fillna(0)

	# 诊断：统计命中情况
	pos_total = len(known)
	pred_total = len(pred)
	overlap = labels["label"].sum()
	print(f"已验证正例={pos_total}, 预测边={pred_total}, 命中的正例数={int(overlap)}")
	if overlap == 0:
		missing = known.loc[~known["key_ud"].isin(pred["key_ud"])]
		print("提示：预测表里没有命中任何正例，可能是以下原因：")
		print("1) 名称或分隔符不一致；2) 预测表仅跨层，正例不在其中；3) 仍存在方向/命名差异")
		print("示例前5条未命中的正例：")
		print(missing.head(5))

	# 关联得分（用无序 key_ud）
	labels = labels.merge(pred[["key_ud", "score"]], on="key_ud", how="left").fillna(0)

	y_true = labels["label"].values
	y_score = labels["score"].values

	# 只有单一类别时不能计算 ROC-AUC/PR-AUC，给出提示
	if len(pd.unique(y_true)) < 2:
		print("警告：y_true 只有单一类别，无法计算 ROC-AUC/PR-AUC。请检查标签或补充负样本。")
	else:
		roc = roc_auc_score(y_true, y_score)
		aupr = average_precision_score(y_true, y_score)
		print(f"ROC-AUC = {roc:.4f}, PR-AUC = {aupr:.4f}")

		# 仅在 top-k（默认为500）上计算 AUC/PR
		k_auc = min(500, len(labels))
		topk_for_auc = labels.sort_values("score", ascending=False).head(k_auc)
		if len(pd.unique(topk_for_auc["label"])) < 2:
			print(f"警告：top{k_auc} 内仅单一类别，无法计算 ROC-AUC/PR-AUC@top{k_auc}")
		else:
			roc_top = roc_auc_score(topk_for_auc["label"], topk_for_auc["score"])
			aupr_top = average_precision_score(topk_for_auc["label"], topk_for_auc["score"])
			print(f"ROC-AUC@top{k_auc} = {roc_top:.4f}, PR-AUC@top{k_auc} = {aupr_top:.4f}")

	# precision@k / recall@k（例如 k=500，但不超过样本数）
	k = min(500, len(labels))
	topk = labels.sort_values("score", ascending=False).head(k)
	tp = topk["label"].sum()
	precision_k = tp / k if k else 0.0
	recall_k = tp / labels["label"].sum() if labels["label"].sum() else 0.0
	print(f"precision@{k} = {precision_k:.4f}, recall@{k} = {recall_k:.4f}")


if __name__ == "__main__":
	main()