# WaifuDiffusion / PixAI Tagger

[WaifuDiffusion (WD)](https://huggingface.co/SmilingWolf) 系タガーと、新しい
[PixAI v0.9 ONNX](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx)
タガーをローカルで動かす Gradio アプリです。
[SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger)
の Hugging Face Space を fork し、PixAI 対応とローカル `./models/`
キャッシュへのオンデマンド DL を追加したものです。

## 対応モデル

| 系統 | モデル |
| --- | --- |
| PixAI | `pixai-tagger-v0.9 (ONNX)` |
| WD v3 | `wd-eva02-large-tagger-v3`, `wd-vit-large-tagger-v3`, `wd-swinv2-tagger-v3`, `wd-convnext-tagger-v3`, `wd-vit-tagger-v3` |
| WD v2 | `wd-v1-4-moat-tagger-v2`, `wd-v1-4-swinv2-tagger-v2`, `wd-v1-4-convnext-tagger-v2`, `wd-v1-4-convnextv2-tagger-v2`, `wd-v1-4-vit-tagger-v2` |
| IdolSankaku | `idolsankaku-swinv2-tagger-v1`, `idolsankaku-eva02-large-tagger-v1` |

PixAI ではキャラクタータグから `selected_tags.csv` の `ips` カラムを介して
著作権 / IP タグを解決します。

## セットアップ

```bash
pip install -r requirements.txt
python app.py
```

Gradio が表示する URL(デフォルト: <http://127.0.0.1:7860>)を開いてください。

モデルが初めて選択された際に、対応する `model.onnx` と `selected_tags.csv`
が Hugging Face から `./models/<dir>/` 配下にダウンロードされ、以降は
キャッシュが使われます。事前一括 DL は行いません。手動で配置したい場合は
該当サブディレクトリに 2 ファイルを置いてください。

GPU を使う場合は `onnxruntime` の代わりに `onnxruntime-gpu` を入れて
CUDA / DirectML execution provider を有効にしてください。

## CLI オプション

```
--score-slider-step           デフォルト 0.05
--score-general-threshold     デフォルト 0.35
--score-character-threshold   デフォルト 0.85
```

## クレジット

- 元 Space と WD タガーモデル: [SmilingWolf](https://huggingface.co/SmilingWolf)
- PixAI v0.9 ONNX 変換: [deepghs](https://huggingface.co/deepghs)
