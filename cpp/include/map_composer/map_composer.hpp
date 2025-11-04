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

struct map_info {
    Cloud map;               // raw map points
    Cloud landmark;          // optional: not used in this minimal ICP-based impl
    std::vector<int> next_no; // adjacency: indices of neighbor maps
};

class MapComposer {
public:
    MapComposer() = default;

    // Load PCD maps (expects files like map1.pcd, map2.pcd, ...)
    bool load_pcds(const std::vector<std::string>& pcd_paths);

    // Load adjacency matrix CSV (square matrix, 0/1 entries)
    bool load_adjacency_csv(const std::string& csv_path);

    // Compose transforms to reference (map index 0) using pairwise ICP along adjacency.
    // Returns true if all pairwise ICP steps converged.
    bool composeMaps();

    // Accessors
    const std::vector<map_info>& maps() const { return m_map_info; }
    const std::vector<Eigen::Isometry2d>& transforms() const { return m_transform; }

    // Utility: apply transform to a cloud (2D XY, Z kept as is); result in XY plane.
    static Cloud transformCloud2D(const Cloud& in, const Eigen::Isometry2d& iso);

private:
    // Estimate rigid transform from src->dst using PCL ICP in 2D (XY only, Z assumed 0).
    // Returns pair (converged, T_4x4) where T maps src to dst.
    static std::pair<bool, Eigen::Matrix4f> icp2D(const Cloud::ConstPtr& src, const Cloud::ConstPtr& dst,
                                                  double max_corr = 0.3, int max_iter = 80, double trans_eps = 1e-6,
                                                  double fitness_eps = 1e-6);

    // Convert 4x4 (XY-plane) to 2D isometry
    static Eigen::Isometry2d toIso2D(const Eigen::Matrix4f& T);

    // Compose A->B then B->C to get A->C
    static Eigen::Isometry2d composeIso2D(const Eigen::Isometry2d& ab, const Eigen::Isometry2d& bc);

    // Invert 2D isometry
    static Eigen::Isometry2d invertIso2D(const Eigen::Isometry2d& iso);

private:
    std::vector<map_info> m_map_info;
    std::vector<Eigen::Isometry2d> m_transform; // per-map transform to reference (index 0)
    std::vector<std::vector<int>> m_adj;        // adjacency list
};

} // namespace mc

