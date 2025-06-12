#!/usr/bin/env python3
"""
ARタグ検出の詳細デバッグ版
パラメータ調整と前処理を試行して検出率を向上
"""

import cv2
import numpy as np
import os


def debug_ar_detection(image_path: str):
    """
    ARタグ検出のデバッグ版
    """
    if not os.path.exists(image_path):
        print(f"エラー: ファイルが見つかりません: {image_path}")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return False
    
    print(f"画像サイズ: {image.shape}")
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # debug_outputフォルダ作成
    os.makedirs("debug_output", exist_ok=True)
    
    # 複数の前処理パターンを試行
    preprocessing_methods = [
        ("original", lambda img: img),
        ("gaussian_blur", lambda img: cv2.GaussianBlur(img, (3, 3), 0)),
        ("median_blur", lambda img: cv2.medianBlur(img, 3)),
        ("clahe", lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)),
        ("histogram_eq", lambda img: cv2.equalizeHist(img)),
        ("threshold", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive_thresh", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
    ]
    
    # 複数のパラメータ設定を試行
    parameter_sets = [
        "default",
        "relaxed",
        "strict"
    ]
    
    detected = False
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for preprocess_name, preprocess_func in preprocessing_methods:
        processed_img = preprocess_func(gray)
        
        # 前処理結果を保存
        cv2.imwrite(f"debug_output/{base_name}_{preprocess_name}.jpg", processed_img)
        
        for param_name in parameter_sets:
            # Aruco設定
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
            parameters = cv2.aruco.DetectorParameters()
            
            # パラメータ調整
            if param_name == "relaxed":
                parameters.adaptiveThreshWinSizeMin = 3
                parameters.adaptiveThreshWinSizeMax = 23
                parameters.adaptiveThreshWinSizeStep = 4
                parameters.adaptiveThreshConstant = 7
                parameters.minMarkerPerimeterRate = 0.01
                parameters.maxMarkerPerimeterRate = 4.0
                parameters.polygonalApproxAccuracyRate = 0.05
                parameters.minCornerDistanceRate = 0.01
                parameters.minDistanceToBorder = 1
                parameters.markerBorderBits = 1
            elif param_name == "strict":
                parameters.adaptiveThreshWinSizeMin = 5
                parameters.adaptiveThreshWinSizeMax = 15
                parameters.adaptiveThreshConstant = 10
                parameters.minMarkerPerimeterRate = 0.1
                parameters.maxMarkerPerimeterRate = 2.0
                parameters.polygonalApproxAccuracyRate = 0.01
                parameters.minCornerDistanceRate = 0.1
            
            # ARタグ検出
            corners, ids, rejected = cv2.aruco.detectMarkers(
                processed_img, aruco_dict, parameters=parameters
            )
            
            if ids is not None and len(ids) > 0:
                print(f"✓ 検出成功: {preprocess_name} + {param_name}")
                print(f"  検出タグ数: {len(ids)}")
                for i, (corner, tag_id) in enumerate(zip(corners, ids)):
                    print(f"  タグ {i+1} (ID: {tag_id[0]}):")
                    corner_points = corner[0]
                    for j, point in enumerate(corner_points):
                        print(f"    座標{j+1}: ({point[0]:.2f}, {point[1]:.2f})")
                
                # 検出結果を可視化
                result_img = image.copy()
                cv2.aruco.drawDetectedMarkers(result_img, corners, ids)
                cv2.imwrite(f"debug_output/{base_name}_detected_{preprocess_name}_{param_name}.jpg", result_img)
                detected = True
                break
            else:
                print(f"✗ 検出失敗: {preprocess_name} + {param_name}")
        
        if detected:
            break
    
    if not detected:
        print("すべての方法で検出に失敗しました")
        
        # リジェクトされたマーカーも確認
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if len(rejected) > 0:
            print(f"リジェクトされた候補: {len(rejected)} 個")
            # リジェクト候補を可視化
            result_img = image.copy()
            for reject in rejected:
                pts = reject.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(result_img, [pts], True, (0, 0, 255), 2)
            cv2.imwrite(f"debug_output/{base_name}_rejected_candidates.jpg", result_img)
            print(f"リジェクト候補を保存: debug_output/{base_name}_rejected_candidates.jpg")
    
    return detected


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python debug_ar_detection.py <image_name>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # パス解決
    if not os.path.exists(input_path):
        image_path = os.path.join("images", input_path)
        if os.path.exists(image_path):
            input_path = image_path
    
    print("=" * 60)
    print("ARタグ検出デバッグ")
    print("=" * 60)
    
    result = debug_ar_detection(input_path)
    
    print("=" * 60)
    print(f"最終結果: {'SUCCESS' if result else 'FAILED'}")
    print("=" * 60)
