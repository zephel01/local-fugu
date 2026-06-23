<h1 align="center">local-fugu</h1>

<p align="center">
  <strong>ローカル LLM で多エージェント・コーディングを回す。<br>SWE-bench を「壊さず・溢れさず・検証して」評価するハーネス付き。</strong>
</p>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python"></a>
  <a href=""><img src="https://img.shields.io/badge/backend-ollama%20%7C%20vllm-orange" alt="backend"></a>
  <a href=""><img src="https://img.shields.io/badge/eval-SWE--bench%20Verified-purple" alt="swebench"></a>
  <a href=""><img src="https://img.shields.io/badge/license-MIT-yellow" alt="license"></a>
</p>

<p align="center">
  <strong>日本語</strong> · <a href="./README.en.md">English</a> · <a href="./ROADMAP.md">ロードマップ</a>
</p>

---

## 何ができるか — 30 秒で

```
                    ユーザの課題 / SWE-bench の issue
                              │
                              ▼
        ┌────────────── local-fugu ──────────────┐
        │  Conductor  (動的ワークフロー生成)       │
        │     │                                   │
        │     ├─ planner   バグ箇所を特定           │
        │     ├─ coder     SEARCH/REPLACE で修正    │
        │     └─ reviewer  検証                     │
        └────────────────┬────────────────────────┘
                         ▼
            検証済みパッチ (difflib + git apply --check + ast)
                         ▼
              SWE-bench 採点 (Docker) / 実行ガイド修復
```

Fugu（Sakana AI）の「複数の専門エージェントを動的に編成する」発想を、**手元の ollama / vLLM だけ**で動かす実験プロジェクト。
さらに、ローカル小型モデルでも **SWE-bench を安全に評価できるハーネス**を備える。

---

## ハイライト

- **多エージェント・パイプライン** — Conductor が `planner → coder → reviewer` の動的ワークフローを生成。エージェント分離でカスケードバイアスを抑制。
- **壊れないパッチ生成** — モデルには SEARCH/REPLACE を出させ、unified diff は `difflib` で生成。`git apply --check`（fuzz なし）＋ `ast.parse` を通過したものだけ採用。**ファイル破壊ゼロ**。
- **コンテキスト圧縮** — 巨大ソースは AST で該当関数のみ逐語抽出（実測 216KB → 約 12KB）。ローカルの小さい窓を溢れさせない。
- **シンボル探索ローカライズ** — issue / 失敗テストのシンボルを `git grep` で定義元まで辿り、編集対象ファイルを特定。
- **実行ガイド修復（opt-in）** — 候補パッチを実際にテスト実行し、失敗 Traceback をコーダに返して再修正。

---

## インストール

```bash
git clone https://github.com/zephel01/local-fugu.git
cd local-fugu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash setup.sh        # 依存 + モデル pull（ollama 前提）
```

---

## 使い方

### 1) 単発のコーディング

```bash
python pipeline.py "スレッドセーフな LRU キャッシュを Python で書いて"
python pipeline.py --output result.py "マージソートをテスト付きで実装"
```

### 2) SWE-bench 評価

```bash
# 予測を生成（3 件だけ試す）
python -m eval.run_swebench --limit 3

# Docker で採点
python eval/run_swebench.py --score-only --output eval/predictions.jsonl

# 実行ガイド修復つき（遅いが resolve を狙う / SWE-bench Docker 必須）
python -m eval.run_swebench --limit 3 --exec-repair
```

---

## 設定 (`config.yaml`)

```yaml
backend: ollama          # "ollama" | "vllm"

agents:
  conductor: "qwen2.5-coder:7b"
  coder:     "qwen3-coder:30b"   # コード特化モデルを推奨
  reviewer:  "qwen2.5-coder:7b"
  planner:   "qwen2.5-coder:7b"

pipeline:
  max_parallel_steps: 2
  timeout_seconds: 600           # 大きめ推奨
```

> **num_ctx に注意**: ollama の既定文脈窓は小さく、超過分は黙って切られる。コーダは 16k 以上で運用すること（Modelfile で `PARAMETER num_ctx 16384`）。

Apple Silicon 向けは `config_mac.yaml` を使用。

---

## SWE-bench ハーネスの設計（要点）

| 仕組み | 目的 |
|---|---|
| **SEARCH/REPLACE → difflib** | 小型モデルに行番号を書かせない（壊れた diff の根絶） |
| **`is_patch_safe`** | `git apply --check` + `ast.parse` を捨てワークツリーで実施。通らなければ空パッチ（非破壊） |
| **検証付きリトライ** | 不一致 SEARCH を具体的に指摘して再生成（最大数回） |
| **AST コンテキスト圧縮** | 該当関数のみ逐語抽出。圧縮→切詰→原文素通しの 3 段で窓あふれ防止 |
| **シンボル探索ローカライズ** | `git grep`（スコープ＋ベース名一致）で定義元ソースを特定 |
| **実行ガイド修復** | テスト実行→Traceback 再投入で「惜しい」を「正解」に近づける |

詳細・既知の限界 → [ROADMAP.md](./ROADMAP.md)

---

## ディレクトリ

```
agents/      Conductor / エージェントプール / ベース
prompts/     各エージェントの指示（md）
eval/        SWE-bench ハーネス（run_swebench / patch_utils / repo_focus / exec_repair …）
merges/      mergekit レシピ（coder/reviewer マージ用）
pipeline.py  パイプライン本体
config*.yaml バックエンド・モデル設定
```

---

## 状態と正直な現状

- ハーネスは健全（破壊ゼロ・タイムアウトなし・検証済みパッチ）。オフライン golden テスト 5 スイートは全 PASS。
- **resolve 率はローカル 30B 級の能力に依存**し、難問では未解決も多い。実力把握には 50〜100 件規模の測定が必要（→ ROADMAP P1）。
- これは研究・実験用プロジェクトであり、ベンチマークスコアの主張を目的としない。

## ライセンス
MIT
