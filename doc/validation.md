# 目的
./dataaフォルダ下にあるテスト用データについての説明用ドキュメントである。

# フォルダ構成
csv/pcdファイル形式で点群の種類がL字型・円型のデータが格納されている。
.data/
├csv/ # csv file
│├circle/ # type:circle
│└L/ # type: L
└pcd # pcd file
 ├circle/ # type:circle
 └L/ # type:L

# ファイル内容
すべての種類でmap1~3があり、隣接行列は.data/adj.csvの通りである。
また、それぞれの変換は以下の通りである。
- map1→map2:平行移動t=(2, 4)、回転:75deg
- map1→map3:平行移動t=(-1, 0)、回転:-20deg

## 座標変換
- 点p=(px, py)を座標変換（平行移動t=(tx, ty)、回転theta）するためのPythonコードは以下とする
``` python
import numpy as np
deg2rad = np.pi/180

def transform_point(p, t, theta):
    R = np.array([[np.cos(theta*deg2rad), -np.sin(theta*deg2rad)],
                  [np.sin(theta*deg2rad), np.cos(theta*deg2rad)]])
    trans_p = np.dot(R, p.reshape([-1, 1])).reshape(-1) + t
    return trans_p
```