# P1: 分業の効果測定 — 実装計画 v1

最終更新: 2026-06-25 / ブランチ: `feature/agent-ablation`

## 結論（測りたい問い）
ハーネス（clone / symbol+test ローカライズ / AST圧縮 / SEARCH→difflib / `is_patch_safe` / retry）を
一定に保ったとき、**planner と reviewer を足すことは coder 単独より SWE-bench resolve を上げるのか**。
OpenFugu の「orchestration は単一最良に勝つか」のローカル分業版。

## アーム（同一 instance を全アームに通す = ペア比較）
- **A: coder のみ** — symbol+test localize → coder の SEARCH/REPLACE + retry。planner/reviewer なし。
- **B: planner+coder** — planner の推論と localize hint を追加。reviewer なし。
- **C: full（配線済 reviewer）** — planner→coder→**reviewer の出力を SEARCH/REPLACE として再検証し、安全な非空パッチなら採用**（不可なら coder パッチ維持）。
  ※重要な発見: リポ本体の `run_fixed_workflow` では reviewer 出力が提出パッチに反映されていない（装飾）。本 ablation の C はこれを配線して reviewer の真の価値を測る。

## モード（両方測る）
- **pure（分業の純化）**: localize を全アーム共通の harness 扱い（symbol+test 固定）、`temperature=0`。
  変数を「どのエージェントが推論するか」だけに限定。
- **real（実運用に忠実）**: 各アームが自然な振る舞い（B/C は planner 由来 hint も localize に使用、温度は config 既定）。

### 条件マトリクス（最大6条件 × N=100）
| 条件 | アーム | localize | temp |
|---|---|---|---|
| A_pure | coder | symbol+test 固定 | 0 |
| B_pure | planner+coder | symbol+test 固定 | 0 |
| C_pure | full | symbol+test 固定 | 0 |
| A_real | coder | symbol+test（planner なしなので pure と実質同一） | config |
| B_real | planner+coder | + planner hint | config |
| C_real | full | + planner hint | config |

> コスト節約メモ: `A_real ≈ A_pure`（温度のみ差）。runner は `--conditions` で取捨可能にし、
> まず A_pure/B_pure/C_pure を回してから real 系を判断する運用も可。

## 統計設計（CONFIDENCE: HIGH）
- ローカル小型モデルは resolve 率が低く N も小さい → **対応あり（ペア）**で見る。
- アーム間の per-instance resolve 差分を **McNemar 検定**（B vs A、C vs B、C vs A）。
- 併せて 95% 信頼区間（Wilson）を resolve 率に付す。素の差だけで結論しない。

## 指標 / 条件
resolve 率・valid patch 率・**PASS_TO_PASS 退行数**・wall-clock/件・生成トークン/件。

## 実装方針（CLAUDE.md 順守）
- 既存リポは**一切編集しない**。`_OUTPUTS/local-fugu-ablation/` に新規モジュールを作り、
  既存ハーネス部品（`eval.repo_context` / `eval.repo_focus` / `eval.patch_utils` / `eval.swe_prompt` / `agents`）を import。
- `run_fixed_workflow` は planner/coder/reviewer がハードコードのため、**エージェント集合をパラメータ化した派生版**を新規作成（元コードを複製せず可能な限り部品を再利用）。
- `feature/agent-ablation` ブランチで作業。main へ直 push しない。

### 成果物
```
_OUTPUTS/local-fugu-ablation/
  run_ablation.py        # パラメータ化 runner（--arm A|B|C --mode pure|real --limit N）
  compare.py             # predictions_*.jsonl + swebench report → McNemar/CI/表
  predictions_{cond}.jsonl
  report_2026MMDD.md     # 比較表 + 検定 + 所見
  PLAN.md (本書)
```

### 検証ステップ（必須）
1. **オフライン golden**: 既存 5 スイートが緑のまま（runner が部品を壊していない）。
2. **パイロット N=20**: 破壊ゼロ・valid patch 率が現行と整合・各アームが想定どおり起動。
3. McNemar の実装をダミーデータで単体検証（既知の分割表で p 値一致）。
4. 本測定 N=100 は GPU 機で実行（generation 直列 ~5h + Docker 採点 max_workers=4 で ~4–5h ≈ 一晩）。

## 想定コスト（正直値）
6条件 × 100件 = 600 instance-run。Docker 採点が支配（~100s/件、4並列で実時間 ~4–5h）。
real 系を削れば 3条件 ×100 に半減可能。
