#!/usr/bin/env python3
"""
ARタグ認識テストプログラム
画像からARタグを検出し、4点の座標が取得できるかをテストする

使用方法:
python ar_tag_detector.py <image_path>
"""

import cv2
import numpy as np
import sys
import os


class ARTagDetector:
    """ARタグ検出クラス"""
    
    def __init__(self):
        """
        初期化
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters()
    
    def detect_ar_tags(self, image: np.ndarray, verbose: bool = False) -> tuple:
        """
        ARタグを検出する
        
        Args:
            image: 入力画像
            verbose: 詳細出力フラグ
            
        Returns:
            (検出成功フラグ, コーナー座標リスト, ID リスト)
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ARタグ検出
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.parameters
        )
        
        success = ids is not None and len(ids) > 0
        
        if verbose:
            if success:
                print(f"ARタグを {len(ids)} 個検出しました")
            else:
                print("ARタグが検出されませんでした")
        
        return success, corners, ids
    
    def draw_detected_tags(self, image: np.ndarray, corners: list, ids: list) -> np.ndarray:
        """
        検出されたARタグを画像に描画
        
        Args:
            image: 入力画像
            corners: コーナー座標リスト
            ids: ID リスト
            
        Returns:
            描画済み画像
        """
        output_image = image.copy()
        
        if len(corners) > 0:
            # ARタグの輪郭を描画
            cv2.aruco.drawDetectedMarkers(output_image, corners, np.array(ids))
            
            # 各コーナーに番号を描画
            for i, corner in enumerate(corners):
                corner_points = corner[0].astype(int)
                for j, point in enumerate(corner_points):
                    cv2.circle(output_image, tuple(point), 5, (0, 255, 0), -1)
                    cv2.putText(output_image, str(j), 
                              (point[0]+10, point[1]+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # タグIDを中央に描画
                center = np.mean(corner_points, axis=0).astype(int)
                cv2.putText(output_image, f"ID:{ids[i]}", 
                          tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_image


def test_ar_detection(image_path: str, save_result: bool = True, verbose: bool = True) -> bool:
    """
    ARタグ検出のテスト関数
    
    Args:
        image_path: 画像ファイルのパス
        save_result: 結果画像を保存するかどうか
        verbose: 詳細出力フラグ
        
    Returns:
        検出成功フラグ
    """
    # 画像読み込み
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return False
    
    if verbose:
        print(f"画像を読み込みました: {image_path}")
        print(f"画像サイズ: {image.shape}")
    
    # ARタグ検出器を作成
    detector = ARTagDetector()
    
    # ARタグ検出
    success, corners, ids = detector.detect_ar_tags(image, verbose=verbose)
    
    # 結果出力
    if success:
        print(f"✓ ARタグ検出成功: {len(ids)} 個のタグを検出")
        for i, (corner, tag_id) in enumerate(zip(corners, ids)):
            print(f"  タグ{i+1} (ID: {tag_id[0]}):")
            corner_points = corner[0]
            for j, point in enumerate(corner_points):
                print(f"    コーナー{j+1}: ({point[0]:.1f}, {point[1]:.1f})")
    else:
        print("✗ ARタグ検出失敗: タグが見つかりませんでした")
    
    # 結果画像を保存
    if save_result:
        # debug_outputフォルダが存在しない場合は作成
        os.makedirs("debug_output", exist_ok=True)
        
        output_image = detector.draw_detected_tags(image, corners, ids)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join("debug_output", f"{base_name}_detected.jpg")
        cv2.imwrite(output_path, output_image)
        print(f"結果画像を保存しました: {output_path}")
    
    return success


def main():
    """メイン関数"""
    if len(sys.argv) != 2:
        print("使用方法: python ar_tag_detector.py <image_name>")
        print("例: python ar_tag_detector.py test.jpg")
        print("例: python ar_tag_detector.py images/test.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # パス解決（imagesディレクトリからの相対パスに対応）
    if not os.path.exists(input_path):
        image_path = os.path.join("images", input_path)
        if os.path.exists(image_path):
            input_path = image_path
    
    print("=" * 50)
    print("ARタグ検出テストプログラム")
    print("=" * 50)
    
    # テスト実行
    success = test_ar_detection(input_path, save_result=True, verbose=True)
    
    print("=" * 50)
    print(f"最終結果: {'SUCCESS' if success else 'FAILED'}")
    print("=" * 50)
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
