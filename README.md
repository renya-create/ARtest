# ARタグ検出テストプログラム

5x5のArucoタグ（DICT_5X5_250）を検出し、4点座標の取得可否をテストする。

## セットアップ

```bash
# 仮想環境を作成・有効化
python3 -m venv .venv
source .venv/bin/activate

# 必要なライブラリをインストール
pip install -r requirements.txt
```

## フォルダ構成

```
ARtest/
├── images/              # 画像ファイルをここに配置
│   ├── test.png
│   └── test2.png
├── debug_output/        # 検出結果画像の出力先
│   └── *_detected.jpg
├── ar_simple_test.py    # シンプルテスト（True/False判定）
├── ar_tag_detector.py   # 詳細テスト（可視化付き）
├── requirements.txt     # 依存ライブラリ
└── .gitignore          # Git管理除外設定
```

## 使い方

### コマンドライン

```bash
# imagesフォルダ内の画像をテスト
python ar_simple_test.py test.jpg

# 詳細テスト（可視化付き）
python ar_tag_detector.py test.jpg

# バッチテスト（imagesフォルダ全体）
python ar_simple_test.py --batch
```

### Pythonコード

```python
from ar_simple_test import test_ar_tag_detection

# 基本テスト
result = test_ar_tag_detection("images/test.jpg")
print(result)  # True or False

# 詳細付きテスト
result = test_ar_tag_detection("images/test.jpg", show_details=True)
```

## メソッド

### `test_ar_tag_detection(image_path, show_details=False)`

画像からARタグを検出し、4点座標の取得可否を返す。

**引数:**
- `image_path`: 画像ファイルパス
- `show_details`: 詳細表示フラグ

**戻り値:** 
- `bool`: 検出成功時True、失敗時False

### `batch_test_images(image_dir="images", extensions=['.jpg', '.jpeg', '.png', '.bmp'])`

ディレクトリ内の画像を一括テストし、成功率を表示する。

**引数:**
- `image_dir`: 画像ディレクトリパス（デフォルト: images）
- `extensions`: 対象拡張子リスト

## 出力

- **シンプルテスト**: コンソールにTrue/False結果
- **詳細テスト**: コンソールに座標情報 + `debug_output/`フォルダに可視化画像
