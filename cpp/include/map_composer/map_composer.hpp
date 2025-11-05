#pragma once

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace mc {

using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;

/**
 * @brief 単一地図に関する生データと隣接情報を保持する構造体。
 */
struct map_info {
    Cloud map;                 ///< 点群データ(生値)
    Cloud landmark;            ///< 予備: 本実装では未使用だが互換性のため残す
    std::vector<int> next_no;  ///< 隣接地図のインデックス一覧
};

/**
 * @brief 点群マップを2D平面上で合成するユーティリティクラス。
 */
class MapComposer {
public:
    MapComposer() = default;

    /**
     * @brief PCDファイル群を読み込み、内部バッファへ展開する。
     * @param pcd_paths 読み込むファイルパスの配列 (map1.pcd 等を想定)
     * @return 読み込みに成功した場合は true
     */
    bool load_pcds(const std::vector<std::string>& pcd_paths);

    /**
     * @brief 隣接行列CSVを読み込み、地図間の接続情報を構築する。
     * @param csv_path 正方行列形式のCSVファイルパス
     * @return 正常に構築できた場合は true
     */
    bool load_adjacency_csv(const std::string& csv_path);

    /**
     * @brief 隣接情報を辿って各地図を基準座標系へ整合させる。
     * @return すべての推定が成功した場合は true
     */
    bool composeMaps();

    const std::vector<map_info>& maps() const { return m_map_info; }
    const std::vector<Eigen::Isometry2d>& transforms() const { return m_transform; }

    /**
     * @brief 2次元等距変換を適用し、XY平面で点群を変換する。
     * @param in 元の点群
     * @param iso 適用する2D変換
     * @return 変換後の点群
     */
    static Cloud transformCloud2D(const Cloud& in, const Eigen::Isometry2d& iso);

private:
    /**
     * @brief PCL の ICP を用いて src→dst の剛体変換を推定する。
     */
    static std::pair<bool, Eigen::Matrix4f> icp2D(const Cloud::ConstPtr& src, const Cloud::ConstPtr& dst,
                                                  double max_corr = 0.3, int max_iter = 80, double trans_eps = 1e-6,
                                                  double fitness_eps = 1e-6);

    /**
     * @brief 4x4同次変換行列を2D等距変換へ射影する。
     */
    static Eigen::Isometry2d toIso2D(const Eigen::Matrix4f& T);

    /**
     * @brief 2つの等距変換を合成し、A→B→C の連鎖を A→C へまとめる。
     */
    static Eigen::Isometry2d composeIso2D(const Eigen::Isometry2d& ab, const Eigen::Isometry2d& bc);

    /**
     * @brief 2D等距変換の逆変換を計算する。
     */
    static Eigen::Isometry2d invertIso2D(const Eigen::Isometry2d& iso);

    /**
     * @brief map_info::next_no から内部隣接リストを再構築する。
     */
    void rebuildAdjacencyFromNextNo();

private:
    std::vector<map_info> m_map_info;
    std::vector<Eigen::Isometry2d> m_transform; ///< 基準地図(0番)への変換
    std::vector<std::vector<int>> m_adj;        ///< 隣接リスト
};

} // namespace mc
