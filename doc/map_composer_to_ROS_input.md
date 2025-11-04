# 目的
- マップ合成のPython実装をpcl::PointCloud<pcl::PointXYZI>を用いたc++実装に変更する。

# 変更要件
- メインの実装は"bool composeMaps()"に実装する（出力はマップ合成の結果が収束したかどうかを表すbool値とし、入力はなし（メンバ変数で実装））
- コードでは以下の定義を用いる
```cpp
using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;
```
- マップの情報（点群、隣接マップ番号）は以下の構造体map_infoのベクトルであるメンバ変数”std::vector<map_info> m_map_info”で所有している。
```cpp
typedef struct{
    Cloud map;
    Cloud landmark;
    std::vector<int> next_no;
}map_info;
```
- マップ合成の結果(基準マップへの変換)は2次元の同次変換行列のベクトルであるメンバ変数”std::vector<Eigen::Isometry2d> m_transform”に保存する
- 動作検証ではマップはpclを使用して、pcdファイルを読み込むようにすること。
