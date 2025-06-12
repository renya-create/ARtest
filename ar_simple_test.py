#!/usr/bin/env python3
"""
簡単なARタグ検出テスト
画像を入力として、ARタグの4点座標が取得できたかをtrue/falseで返す

使用方法:
from ar_simple_test import test_ar_tag_detection
result = test_ar_tag_detection("image.jpg")
print(result)  # True or False
"""

import cv2
import numpy as np
import os


def test_ar_tag_detection(image_path: str, show_details: bool = False) -> bool:
    """
    ARタグ検出の簡単なテスト関数（複数パラメータ自動試行）
    
    Args:
        image_path: 画像ファイルのパス
        show_details: 詳細を表示するかどうか
        
    Returns:
        bool: ARタグの4点座標が取得できた場合True、そうでなければFalse
    """
    # 画像ファイルの存在確認
    if not os.path.exists(image_path):
        if show_details:
            print(f"エラー: ファイルが見つかりません: {image_path}")
        return False
    
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        if show_details:
            print(f"エラー: 画像を読み込めません: {image_path}")
        return False
    
    # 5x5のAruco辞書を設定
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 複数のパラメータ設定を順次試行
    parameter_configs = [
        {
            "name": "default",
            "params": {}  # デフォルト設定
        },
        {
            "name": "relaxed", 
            "params": {
                "adaptiveThreshWinSizeMin": 3,
                "adaptiveThreshWinSizeMax": 23,
                "adaptiveThreshWinSizeStep": 4,
                "adaptiveThreshConstant": 7,
                "minMarkerPerimeterRate": 0.01,
                "maxMarkerPerimeterRate": 4.0,
                "polygonalApproxAccuracyRate": 0.05,
                "minCornerDistanceRate": 0.01,
                "minDistanceToBorder": 1
            }
        },
        {
            "name": "strict",
            "params": {
                "adaptiveThreshWinSizeMin": 5,
                "adaptiveThreshWinSizeMax": 15,
                "adaptiveThreshConstant": 10,
                "minMarkerPerimeterRate": 0.1,
                "maxMarkerPerimeterRate": 2.0,
                "polygonalApproxAccuracyRate": 0.01,
                "minCornerDistanceRate": 0.1
            }
        }
    ]
    
    detected = False
    successful_config = None
    corners = None
    ids = None
    
    for config in parameter_configs:
        # パラメータを設定
        parameters = cv2.aruco.DetectorParameters()
        
        # カスタムパラメータを適用
        for param_name, param_value in config["params"].items():
            if hasattr(parameters, param_name):
                setattr(parameters, param_name, param_value)
        
        # ARタグ検出
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
        
        if ids is not None and len(ids) > 0:
            detected = True
            successful_config = config["name"]
            if show_details:
                print(f"✓ ARタグ検出成功: {len(ids)} 個 ({successful_config}パラメータで検出)")
            break
        elif show_details:
            print(f"✗ {config['name']}パラメータ: 検出なし")
    
    # 結果の詳細表示
    if show_details:
        if detected:
            for i, (corner, tag_id) in enumerate(zip(corners, ids)):
                print(f"  タグ {i+1} (ID: {tag_id[0]}):")
                corner_points = corner[0]
                for j, point in enumerate(corner_points):
                    print(f"    座標{j+1}: ({point[0]:.2f}, {point[1]:.2f})")
        else:
            print("✗ すべてのパラメータでARタグが検出されませんでした")
    
    return detected


def batch_test_images(image_dir: str = "images", extensions: list = ['.jpg', '.jpeg', '.png', '.bmp']):
    """
    指定ディレクトリ内の画像ファイルをバッチテストする
    
    Args:
        image_dir: 画像ディレクトリのパス（デフォルト: images）
        extensions: テスト対象の拡張子リスト
    """
    if not os.path.exists(image_dir):
        print(f"エラー: ディレクトリが見つかりません: {image_dir}")
        return
    
    results = []
    image_files = []
    
    # 画像ファイルを収集
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"画像ファイルが見つかりませんでした: {image_dir}")
        return
    
    print(f"バッチテスト開始: {len(image_files)} 個のファイル")
    print("=" * 60)
    
    success_count = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = test_ar_tag_detection(image_path, show_details=False)
        results.append((image_file, result))
        
        status = "✓ SUCCESS" if result else "✗ FAILED"
        print(f"{image_file:30} : {status}")
        
        if result:
            success_count += 1
    
    print("=" * 60)
    print(f"結果: {success_count}/{len(image_files)} 個の画像でARタグを検出")
    print(f"成功率: {success_count/len(image_files)*100:.1f}%")


def resolve_image_path(path: str) -> str:
    """
    画像パスを解決する（imagesディレクトリからの相対パスに対応）
    
    Args:
        path: 入力パス
        
    Returns:
        解決されたパス
    """
    # 絶対パスまたは相対パスで存在する場合はそのまま
    if os.path.exists(path):
        return path
    
    # imagesディレクトリ内のファイルとして解釈
    image_path = os.path.join("images", path)
    if os.path.exists(image_path):
        return image_path
    
    # 元のパスを返す（エラーハンドリングは呼び出し側で）
    return path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  単一画像テスト: python ar_simple_test.py <image_name>")
        print("    例: python ar_simple_test.py test.jpg")
        print("    例: python ar_simple_test.py images/test.jpg")
        print("  バッチテスト:   python ar_simple_test.py --batch")
        print("    例: python ar_simple_test.py --batch")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        # バッチテスト（imagesディレクトリ）
        batch_test_images("images")
    else:
        # 単一画像テスト
        input_path = sys.argv[1]
        resolved_path = resolve_image_path(input_path)
        result = test_ar_tag_detection(resolved_path, show_details=True)
        print(f"\n最終結果: {result}")
