# -*- coding: utf-8 -*-
"""
实验一（逻辑回归部分）：基于 TF-IDF 与 LogisticRegression 的新闻文本 10 分类
==========================================================================
本脚本与 svm_experiment.py 相互独立，各自完成一种模型的完整实验流程，可单独运行：
  1. 数据加载，按 80/20 分层划分训练/验证集（测试集无标签，保持封存到最终预测）
  2. 文本清洗消融（原始 vs 清洗，以 LR 自身为对照模型）
  3. TF-IDF 特征工程网格（max_features × ngram，再单因素停用词、sublinear_tf；LR 自身为代理）
  4. LR 调参：C∈{0.1,1,10,100}（max_iter=2000 保证收敛，不进网格）
  5. 训练过程损失曲线：saga 求解器 + warm_start 逐迭代记录训练/验证交叉熵损失
  6. 最优模型详细评估：分类报告、混淆矩阵、错误分析；3 个随机种子检查可复现性
  7. 最优配置在全量训练集重训，对无标签测试集预测 ->
     output_lr/predictions.csv，并覆盖课程提交文件 data-Project1/predictions.csv
     （LR 是实验报告综合性能与复杂度选定的提交模型）

可复现性：SEED 固定贯穿划分/模型；TF-IDF 只在训练子集 fit；验证集只用于调参。
运行：    python lr_experiment.py     （在 data-Project1 目录下）
输出：    output_lr/figures/*.png, output_lr/tables/*, output_lr/predictions.csv, output_lr/summary.json
"""

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import train_test_split

# ----------------------------- 全局配置 -----------------------------
SEED = 42                       # 固定随机种子，保证可复现
VAL_SIZE = 0.2                  # 验证集比例（从训练集中分层划出）
CLASSES = np.arange(10)
DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "output_lr"
FIG_DIR = OUT / "figures"
TBL_DIR = OUT / "tables"
for d in (OUT, FIG_DIR, TBL_DIR):
    d.mkdir(exist_ok=True)

np.random.seed(SEED)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
ALL_ROWS = []       # 汇总所有评估记录（配置、验证集指标、耗时）
SUMMARY = {"model_family": "LogisticRegression"}

# 断点续跑：每完成一次评估立即落盘，重跑时按 (model, note) 命中缓存，跳过已完成的训练
CKPT = TBL_DIR / "checkpoint_rows.jsonl"
if CKPT.exists():
    with open(CKPT, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line:
                ALL_ROWS.append(json.loads(_line))
    print(f"[续跑] 从缓存载入 {len(ALL_ROWS)} 条已完成的评估记录")


def log(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fig_save(fig, name: str):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  图已保存: figures/{name}")


# ----------------------------- 数据加载与划分 -----------------------------
HEADER_KEYS = (
    r"^(From|Subject|Organization|Lines|Reply-To|NNTP-Posting-Host|Nntp-Posting-Host|"
    r"Distribution|X-Newsreader|Article-I\.D\.|Expires|References|Sender|Followup-To|"
    r"Newsgroups|Path|Date|Message-ID|In-Reply-To|Xref|Summary|Keywords|Supersedes|"
    r"Originator|Apparently-To|Return-Path|To|Cc|Gcc|Version|Nf-Id|Nf-From)\s*:"
)


def clean_text(text: str) -> str:
    """去除正文以外的信息（邮件头块、邮箱地址、非字母符号），保留正文词。"""
    head, sep, rest = text.partition("\n\n")
    # 开头到第一个空行之间若出现 >=2 个邮件头字段，则判定为头块并丢弃
    if sep and len(re.findall(HEADER_KEYS, head, flags=re.M)) >= 2:
        text = rest
    text = re.sub(r"\S+@\S+", " ", text)      # 邮箱地址
    text = re.sub(r"[^A-Za-z\s]", " ", text)  # 标点/数字等非字母符号
    return re.sub(r"\s+", " ", text).strip()


def load_and_split():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    X_all = train_df["text"].astype(str).tolist()
    y_all = train_df["target"].values
    X_test = test_df["text"].astype(str).tolist()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, stratify=y_all, random_state=SEED)

    # 清洗是无参数的规则变换，不涉及从数据学习的统计量，可对各子集独立应用
    X_tr_c = [clean_text(t) for t in X_tr]
    X_va_c = [clean_text(t) for t in X_va]
    log(f"数据加载完成: 全量训练 {len(X_all)}, 划分后 训练 {len(X_tr)} / 验证 {len(X_va)}, "
        f"测试(无标签) {len(X_test)}")
    return X_all, y_all, X_test, X_tr, X_va, y_tr, y_va, X_tr_c, X_va_c


def evaluate(estimator, X_tr, y_tr, X_va, y_va, model: str, note: str):
    """拟合 + 验证集评估，统一记录 accuracy / macro-F1 / 训练与预测耗时。"""
    est = clone(estimator)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    t0 = time.time()
    pred = est.predict(X_va)
    pred_s = time.time() - t0
    acc = accuracy_score(y_va, pred)
    f1m = f1_score(y_va, pred, average="macro")
    row = dict(model=model, note=note,
               val_acc=round(acc, 4), val_macro_f1=round(f1m, 4),
               fit_s=round(fit_s, 2), pred_s=round(pred_s, 3))
    log(f"  {model:<12s} {note:<34s} acc={acc:.4f}  macroF1={f1m:.4f}  fit={fit_s:.1f}s")
    return row, est


def cached_eval(estimator, X_tr, y_tr, X_va, y_va, model: str, note: str):
    """带断点续跑缓存的评估：命中 (model, note) 则直接复用历史记录，不重复训练。"""
    for r in ALL_ROWS:
        if r["model"] == model and r["note"] == note:
            log(f"  (缓存命中) {model:<12s} {note:<34s} acc={r['val_acc']}")
            return r, None
    row, est = evaluate(estimator, X_tr, y_tr, X_va, y_va, model, note)
    ALL_ROWS.append(row)
    with open(CKPT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row, est


def make_lr(C=1.0, seed=SEED):
    # max_iter=2000：默认 100 次会触发 ConvergenceWarning，属收敛保障而非调参对象
    return LogisticRegression(C=C, max_iter=2000, random_state=seed)


# ----------------------------- 主流程 -----------------------------
def main():
    (X_all, y_all, X_test, X_tr, X_va, y_tr, y_va, X_tr_c, X_va_c) = load_and_split()
    SUMMARY["split"] = dict(n_train_full=len(X_all), n_train_part=len(X_tr),
                            n_val=len(X_va), n_test=len(X_test), val_size=VAL_SIZE,
                            seed=SEED)

    # ===== 步骤 A：文本清洗消融（LR 自身为对照模型） =====
    log("== A. 文本清洗消融: 原始 vs 清洗（LR C=1, 默认 TF-IDF）==")
    vec0 = dict(max_features=5000, ngram_range=(1, 1))
    abl_rows = []
    for tag, tr, va in [("原始文本", X_tr, X_va), ("清洗后", X_tr_c, X_va_c)]:
        v = TfidfVectorizer(**vec0)
        Xtr = v.fit_transform(tr)          # 只在训练集 fit
        Xva = v.transform(va)
        r, _ = cached_eval(make_lr(), Xtr, y_tr, Xva, y_va, "LR", tag)
        abl_rows.append(r)
    pd.DataFrame(abl_rows).to_csv(TBL_DIR / "ablation_clean.csv", index=False)
    abl_map = {r["note"]: r for r in abl_rows}
    use_clean = abl_map["清洗后"]["val_acc"] > abl_map["原始文本"]["val_acc"]
    SUMMARY["cleaning_ablation"] = dict(rows=abl_rows, use_clean=use_clean)
    log(f"  结论: 原始 {abl_rows[0]['val_acc']} vs 清洗 {abl_rows[1]['val_acc']} "
        f"-> 采用{'清洗后' if use_clean else '原始'}文本")
    if use_clean:
        X_tr_use, X_va_use = X_tr_c, X_va_c
    else:
        X_tr_use, X_va_use = X_tr, X_va

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.bar([r["note"] for r in abl_rows], [r["val_acc"] for r in abl_rows],
           width=0.45, color=["#8da0cb", "#fc8d62"])
    ax.set_ylim(0.7, None)
    for i, r in enumerate(abl_rows):
        ax.text(i, r["val_acc"] + 0.004, f"{r['val_acc']:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("验证集 Accuracy")
    ax.set_title(f"LR 文本清洗消融（默认TF-IDF, seed={SEED}）")
    ax.grid(axis="y", alpha=0.3)
    fig_save(fig, "figL1_ablation_clean.png")

    # ===== 步骤 B：TF-IDF 特征网格（LR 自身为代理，逐因素推进） =====
    log("== B. TF-IDF 特征工程网格（代理模型: LR） ==")

    def vec_eval(cfg, note):
        v = TfidfVectorizer(**cfg)
        Xtr = v.fit_transform(X_tr_use)
        Xva = v.transform(X_va_use)
        r, _ = cached_eval(make_lr(), Xtr, y_tr, Xva, y_va, "LR(proxy)", note)
        return r["val_acc"], len(v.vocabulary_)

    grid_rows = []
    for mf in [5000, 20000, 50000]:
        for ng in [(1, 1), (1, 2)]:
            cfg = dict(max_features=mf, ngram_range=ng)
            acc, vocab = vec_eval(cfg, f"mf={mf},ng={ng[1]}")
            grid_rows.append(dict(max_features=mf, ngram=f"{ng[0]}-{ng[1]}gram",
                                  val_acc=round(acc, 4), vocab=vocab))
    gdf = pd.DataFrame(grid_rows)
    best_g = gdf.loc[gdf["val_acc"].idxmax()]
    ng_best = (1, 2) if best_g["ngram"] == "1-2gram" else (1, 1)
    vec_cfg = dict(max_features=int(best_g["max_features"]), ngram_range=ng_best)

    # 单因素 1：停用词
    for sw in [None, "english"]:
        acc, _ = vec_eval(dict(vec_cfg, stop_words=sw),
                          f"stopwords={'english' if sw else 'None'}")
        grid_rows.append(dict(max_features=vec_cfg["max_features"], ngram=best_g["ngram"],
                              val_acc=round(acc, 4), vocab=np.nan,
                              factor=f"stopwords={'english' if sw else 'None'}"))
    sw_rows = [r for r in grid_rows if r.get("factor", "").startswith("stopwords")]
    if max(sw_rows, key=lambda r: r["val_acc"])["factor"] == "stopwords=english":
        vec_cfg["stop_words"] = "english"

    # 单因素 2：sublinear_tf
    for stf in [False, True]:
        acc, _ = vec_eval(dict(vec_cfg, sublinear_tf=stf), f"sublinear={stf}")
        grid_rows.append(dict(max_features=vec_cfg["max_features"], ngram=best_g["ngram"],
                              val_acc=round(acc, 4), vocab=np.nan,
                              factor=f"sublinear={stf}"))
    stf_rows = [r for r in grid_rows if r.get("factor", "").startswith("sublinear")]
    if max(stf_rows, key=lambda r: r["val_acc"])["factor"] == "sublinear=True":
        vec_cfg["sublinear_tf"] = True

    pd.DataFrame(grid_rows).to_csv(TBL_DIR / "tfidf_grid.csv", index=False)
    SUMMARY["tfidf"] = dict(final_cfg={k: str(v) for k, v in vec_cfg.items()}, grid=grid_rows)
    log(f"  TF-IDF 最终配置: {vec_cfg}")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for ng, sub in gdf.groupby("ngram"):
        ax.plot(sub["max_features"], sub["val_acc"], "o-", label=ng)
    ax.set_xscale("log")
    ax.set_xticks([5000, 20000, 50000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("max_features（log 轴）")
    ax.set_ylabel("验证集 Accuracy (LR)")
    ax.set_title("TF-IDF 参数网格（LR 代理）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_save(fig, "figL2_tfidf_grid.png")

    # 用最终 TF-IDF 配置生成后续调参用的特征
    vectorizer = TfidfVectorizer(**vec_cfg)
    Xtr_tfidf = vectorizer.fit_transform(X_tr_use)   # 只在训练集 fit
    Xva_tfidf = vectorizer.transform(X_va_use)
    log(f"  特征矩阵: 训练 {Xtr_tfidf.shape}, 验证 {Xva_tfidf.shape}, "
        f"稀疏度 {1 - Xtr_tfidf.nnz / (Xtr_tfidf.shape[0] * Xtr_tfidf.shape[1]):.4f}")
    SUMMARY["feature"] = dict(shape=list(Xtr_tfidf.shape), vocab=len(vectorizer.vocabulary_),
                              sparsity=round(1 - Xtr_tfidf.nnz / (Xtr_tfidf.shape[0] * Xtr_tfidf.shape[1]), 5))

    # ===== 步骤 C：LR 调参（C 网格，全部以验证集为准） =====
    log("== C. LogisticRegression: C 网格（max_iter=2000）==")
    lr_rows = []
    for c in [0.1, 1, 10, 100]:
        r, _ = cached_eval(make_lr(c), Xtr_tfidf, y_tr, Xva_tfidf, y_va, "LR", f"C={c}")
        lr_rows.append(r)

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    cs = [0.1, 1, 10, 100]
    ax.plot(cs, [r["val_acc"] for r in lr_rows], "o-", label="LogisticRegression")
    ax.set_xscale("log")
    ax.set_xlabel("C（log 轴）")
    ax.set_ylabel("验证集 Accuracy")
    ax.set_title("正则化参数 C 的影响（其余条件固定）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_save(fig, "figL3_C_curve.png")

    # ===== 步骤 D：最优配置详细评估（分类报告 / 混淆矩阵 / 错误分析） =====
    rdf_main = pd.DataFrame([r for r in ALL_ROWS
                             if r["model"] == "LR" and "=" in r["note"]])
    best_row = rdf_main.loc[rdf_main["val_macro_f1"].idxmax()]
    cand = rdf_main[rdf_main["val_macro_f1"] == best_row["val_macro_f1"]]
    best_row = cand.loc[cand["val_acc"].idxmax()]
    best_c = float(best_row["note"].split("C=")[1])
    SUMMARY["best_model"] = dict(note=best_row["note"], val_acc=best_row["val_acc"],
                                 val_macro_f1=best_row["val_macro_f1"])
    log(f"== D. LR 最优配置: {best_row['note']}  acc={best_row['val_acc']}, "
        f"macroF1={best_row['val_macro_f1']} ==")

    ckpt_best = TBL_DIR / "best_model_ckpt.json"
    saved = None
    if ckpt_best.exists():
        saved = json.loads(ckpt_best.read_text(encoding="utf-8"))
        if saved["note"] != best_row["note"]:
            saved = None
        else:
            y_pred_va = np.array(saved["y_pred_va"])
            log("  (缓存命中) 最优模型验证集预测结果")
    if saved is None:
        best_est = make_lr(best_c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_est.fit(Xtr_tfidf, y_tr)
        y_pred_va = best_est.predict(Xva_tfidf)
        ckpt_best.write_text(json.dumps(dict(note=best_row["note"],
                                             y_pred_va=[int(v) for v in y_pred_va])),
                             encoding="utf-8")
    rep = classification_report(y_va, y_pred_va, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.round(4).to_csv(TBL_DIR / "class_report_best.csv")
    cm = confusion_matrix(y_va, y_pred_va)
    np.savetxt(TBL_DIR / "confusion_matrix_best.csv", cm, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"LR 最优（{best_row['note']}）验证集混淆矩阵")
    fig.colorbar(im, fraction=0.046)
    fig_save(fig, "figL4_confusion.png")

    err_pairs = sorted(((cm[i, j], i, j) for i in range(10) for j in range(10) if i != j),
                       reverse=True)[:5]
    SUMMARY["top_confusions"] = [dict(true=int(i), pred=int(j), n=int(n)) for n, i, j in err_pairs]
    with open(TBL_DIR / "misclassified_examples.txt", "w", encoding="utf-8") as f:
        f.write(f"LR 最优 ({best_row['note']}) 验证集误判样例\n\n")
        shown = 0
        for idx in range(len(y_va)):
            if y_va[idx] != y_pred_va[idx]:
                f.write(f"--- 样例 {idx} | 真实={y_va[idx]} 预测={y_pred_va[idx]} ---\n")
                f.write(X_va_use[idx][:300].replace("\n", " ") + "\n\n")
                shown += 1
                if shown >= 6:
                    break

    # ===== 步骤 E：训练过程损失曲线（saga, 交叉熵损失） =====
    log("== E. 训练过程损失曲线（saga, 交叉熵） ==")
    lc_path = TBL_DIR / "loss_curve.json"
    if lc_path.exists():
        curve = json.loads(lc_path.read_text(encoding="utf-8"))
        log("  (缓存命中) 损失曲线数据")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # warm_start 每次续训 1 个迭代（saga 一个 epoch），从初始权重记录损失下降过程
            lrc = LogisticRegression(C=best_c, max_iter=1, solver="saga",
                                     warm_start=True, random_state=SEED)
            curve = dict(iters=[], train_loss=[], val_loss=[], val_acc=[])
            for _ in range(40):
                lrc.fit(Xtr_tfidf, y_tr)
                curve["iters"].append(len(curve["iters"]) + 1)
                curve["train_loss"].append(
                    log_loss(y_tr, lrc.predict_proba(Xtr_tfidf), labels=CLASSES))
                curve["val_loss"].append(
                    log_loss(y_va, lrc.predict_proba(Xva_tfidf), labels=CLASSES))
                curve["val_acc"].append(accuracy_score(y_va, lrc.predict(Xva_tfidf)))
        with open(lc_path, "w", encoding="utf-8") as f:
            json.dump(curve, f, ensure_ascii=False, indent=1)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(curve["iters"], curve["train_loss"], "o-", ms=3, label="训练损失")
    ax.plot(curve["iters"], curve["val_loss"], "s-", ms=3, label="验证损失")
    ax.set_xlabel("优化迭代次数 (saga, 每次 1 个 epoch)")
    ax.set_ylabel("交叉熵损失 log-loss")
    ax.set_title(f"LogisticRegression (C={best_c}) 训练过程")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(curve["iters"], curve["val_acc"], ".:", color="gray", alpha=0.6)
    ax2.set_ylabel("验证 Accuracy", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    fig_save(fig, "figL5_loss_curve.png")

    # ===== 步骤 F：随机种子检查（3 个种子，验证可复现性） =====
    seed_rows = []
    for sd in [42, 7, 2025]:
        r, _ = cached_eval(make_lr(best_c, seed=sd), Xtr_tfidf, y_tr, Xva_tfidf, y_va,
                           "LR-seed", f"seed={sd}, C={best_c}")
        seed_rows.append(r)
    accs = [r["val_acc"] for r in seed_rows]
    SUMMARY["seed_stability"] = dict(accs=accs, mean=round(float(np.mean(accs)), 4),
                                     std=round(float(np.std(accs)), 4))
    pd.DataFrame(seed_rows).to_csv(TBL_DIR / "lr_seeds.csv", index=False)

    # ===== 步骤 G：最优配置在全量训练集重训 -> 测试集预测 -> 提交 predictions.csv =====
    log("== G. 全量重训并预测测试集 ==")
    X_all_use = [clean_text(t) for t in X_all] if use_clean else X_all
    X_test_use = [clean_text(t) for t in X_test] if use_clean else X_test
    vec_full = TfidfVectorizer(**vec_cfg)
    Xfull = vec_full.fit_transform(X_all_use)      # 词表在全量训练集重新学习
    Xtest = vec_full.transform(X_test_use)
    final_est = make_lr(best_c)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_est.fit(Xfull, y_all)
    log(f"  全量重训完成 ({time.time() - t0:.1f}s), 训练矩阵 {Xfull.shape}, 测试矩阵 {Xtest.shape}")
    preds = final_est.predict(Xtest).astype(int)
    pd.Series(preds).to_csv(OUT / "predictions.csv", index=False, header=False)
    # LR 为报告选定的提交模型：覆盖课程示例 predictions.csv
    pd.Series(preds).to_csv(DATA_DIR / "predictions.csv", index=False, header=False)
    dist = np.bincount(preds, minlength=10)
    SUMMARY["final_test"] = dict(n=len(preds), prediction_distribution=dist.tolist(),
                                 final_fit_s=round(time.time() - t0, 1))
    log(f"  预测分布: {dist.tolist()} -> output_lr/predictions.csv 及提交文件 predictions.csv")

    SUMMARY["rows_all"] = ALL_ROWS
    SUMMARY["finished_at"] = datetime.now().isoformat(timespec="seconds")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=2, default=str)
    pd.DataFrame(ALL_ROWS).to_csv(TBL_DIR / "all_results.csv", index=False)
    log(f"LR 实验全部完成，总耗时 {time.time() - T0:.1f}s。输出目录: {OUT}")


if __name__ == "__main__":
    main()
