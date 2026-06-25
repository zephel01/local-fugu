# RUNBOOK — P1 ablation 実行手順（GPU 機）

前提: ollama 起動・対象モデル pull 済み・SWE-bench Docker 採点が動く環境。
本 runner はリポ本体を編集しない。すべて `_OUTPUTS/local-fugu-ablation/` 配下で動く。

## 0. ブランチ（CLAUDE.md 順守）
```bash
git checkout -b feature/agent-ablation   # main へ直 commit しない
```

## 1. パイロット N=20（破壊ゼロ・起動確認）
まず pure 3 アームを 20 件で。各アームが想定どおり起動し valid patch が出るか確認。
```bash
cd _OUTPUTS/local-fugu-ablation
for ARM in A B C; do
  python run_ablation.py --arm $ARM --mode pure --limit 20 \
      --output preds_${ARM}_pure_n20.jsonl
done
# 期待: A は planner ステップ無し / C は "reviewer (wired)" ログが出る / empty でも破壊ゼロ
```

## 2. 本測定 N=100（6 条件 — A_real≈A_pure なので削れば 5）
```bash
N=100
for MODE in pure real; do
  for ARM in A B C; do
    python run_ablation.py --arm $ARM --mode $MODE --limit $N \
        --output preds_${ARM}_${MODE}.jsonl
  done
done
# コスト節約: A_real は A_pure とほぼ同一（温度のみ差）。省くなら上の real ループから A を除外。
```
> 同一 N=100 instance を全条件に通すため、`--limit 100` は dataset 先頭から決定的に同じ100件を取る。
> 途中再開は `--resume`。

## 3. Docker 採点（条件ごとに run_id を分けて report を衝突させない）
リポの `score-only` は run_id 固定で report が上書きされるため、公式 harness を直接呼ぶ：
```bash
for COND in A_pure B_pure C_pure A_real B_real C_real; do
  python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Verified --split test \
      --predictions_path preds_${COND}.jsonl \
      --run_id ${COND} --max_workers 4 \
      --report_dir reports/
done
# 生成物: reports/local-fugu-<arm>-<mode>.${COND}.json （resolved_ids を含む）
```

## 4. 比較・統計
```bash
python compare.py \
  --cond A_pure:preds_A_pure.jsonl:reports/<...A_pure...>.json \
  --cond B_pure:preds_B_pure.jsonl:reports/<...B_pure...>.json \
  --cond C_pure:preds_C_pure.jsonl:reports/<...C_pure...>.json \
  --cond A_real:preds_A_real.jsonl:reports/<...A_real...>.json \
  --cond B_real:preds_B_real.jsonl:reports/<...B_real...>.json \
  --cond C_real:preds_C_real.jsonl:reports/<...C_real...>.json \
  --pair B_pure:A_pure --pair C_pure:B_pure --pair C_pure:A_pure \
  --pair B_real:A_real --pair C_real:B_real \
  --pair A_real:A_pure \
  --out report_$(date +%Y%m%d).md
```
出力: 各条件の resolve 率 + Wilson 95% CI、ペアごとの McNemar(p)。

## 解釈の指針
- `b - c` が第1アームの純勝ち数。p が小さいほど偶然では説明しにくい差。
- resolve 率が低く N≈100 では CI は広い。一桁の差は「示唆」止まり、断定しない。
- `A_real vs A_pure` はサニティ（温度のみ差 → 大差が出たら温度ノイズが大きい証拠）。

## 既知の制約
- 本 ablation は **exec-repair なし**（分業とは別軸。混ぜると交絡）。
- 採点 report のファイル名は swebench のバージョンで変わる。`reports/` 内の実ファイル名を確認して `--cond` に渡す。
