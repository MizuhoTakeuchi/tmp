# 目的
- 座標系が異なる2次元の点群マップを基準マップの座標系（以下map1を基準マップとする）へ統合する変換（2×2回転行列Rと2次元並進t）を求めるアルゴリズムの要件を記載

# 要件
## 入出力
- 入力
  - `map_files: List[str]`：CSVファイルパス列
  - `adjacency_matrix: np.ndarray`：隣接行列
- 出力
  - `map*_to_map1: np.ndarray`：map*からmap1への変換（2×2回転行列Rと2次元並進t）
  - `map*_in1: np.ndarray`：map1座標系に変換したmap*の点群

## 点群種類
- 点群マップに含まれる点群は以下の2種類
  - L字型：
    - 2本の線分からなり、片方の端点（以下交点と呼ぶ）が一致
    - 特徴量は、交点の座標値（lc=(lcx, lcy)）、線分の長さ(l1, l2)、線分のなす角（ldeg）
    - 線分のなす角は90°とは限らない
  - 円型：
    - 半径0.025mの円であり、距離が近い複数個がグループとなる
    - 特徴量はグループ内の各円の中心点（c=(cx,cy)）
    - 同じグループ内では2m以内に他の円が一つ以上ある

## アルゴリズムフロー概要
1) データ読み込み（csvファイルとする）
2) 各マップの点群をクラスタリング
3) クラスタ毎に点群タイプを判別
4) 隣り合うマップ同士で、クラスタ分けされた点群の特徴量から一致しているクラスタを選択
5) 特徴量が一致するように初期変換R,tを推定
6) 隣接関係に沿って BFS で参照座標系へ変換を連鎖合成
7) ICP(Iterative Closest Point)アルゴリズムで、すべてのmap1への変換をまとめて微調整

## 要求事項
- 点群にノイズが乗っていてもロバストなアルゴリズムとすること
- 一度に与えられるマップに含まれる点群タイプは1種類のみ（L字型と円型が混在することはない）である

## 座標変換
- 点p=(px, py)を座標変換（平行移動t=(tx, ty)、回転theta）するためのコードは以下とする
``` python
import numpy as np
deg2rad = np.pi/180

def transform_point(p, t, theta):
    R = np.array([[np.cos(theta*deg2rad), -np.sin(theta*deg2rad)],
                  [np.sin(theta*deg2rad), np.cos(theta*deg2rad)]])
    trans_p = np.dot(R, p.reshape([-1, 1])).reshape(-1) + t
    return trans_p
```