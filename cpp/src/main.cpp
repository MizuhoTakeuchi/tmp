#include "map_composer/map_composer.hpp"

#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief コマンドラインの使い方を表示する。
 */
static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --pcd-dir <dir> [--adj <path>] [--write-out <dir>]" << std::endl;
}

/**
 * @brief map1.pcd ~ map3.pcd のパスをまとめて生成する。
 */
static std::vector<std::string> default_pcds(const std::string& dir) {
    std::vector<std::string> out;
    for (int i = 1; i <= 3; ++i) {
        fs::path p = fs::path(dir) / (std::string("map") + std::to_string(i) + ".pcd");
        out.push_back(p.string());
    }
    return out;
}

/**
 * @brief 推定した変換の概要を表示する。
 */
static void print_iso_summary(const Eigen::Isometry2d& iso, int idx) {
    double th = std::atan2(iso.linear()(1,0), iso.linear()(0,0)) * 180.0 / M_PI;
    std::cout << "map" << (idx+1) << " -> map1: t=("
              << iso.translation().x() << ", " << iso.translation().y()
              << "), theta_deg=" << th << std::endl;
}

int main(int argc, char** argv) {
    std::string pcd_dir;
    std::string adj_csv = "data/adj.csv";
    std::string out_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pcd-dir" && i+1 < argc) { pcd_dir = argv[++i]; }
        else if (a == "--adj" && i+1 < argc) { adj_csv = argv[++i]; }
        else if (a == "--write-out" && i+1 < argc) { out_dir = argv[++i]; }
        else { /* ignore */ }
    }
    if (pcd_dir.empty()) { usage(argv[0]); return 2; }

    mc::MapComposer composer;
    auto pcds = default_pcds(pcd_dir);
    if (!composer.load_pcds(pcds)) return 1;
    if (!composer.load_adjacency_csv(adj_csv)) return 1;

    bool ok = composer.composeMaps();
    std::cout << (ok ? "composeMaps converged on all edges" : "composeMaps had non-converged edges") << std::endl;

    const auto& T = composer.transforms();
    for (size_t i = 0; i < T.size(); ++i) {
        print_iso_summary(T[i], static_cast<int>(i));
    }

    if (!out_dir.empty()) {
        fs::create_directories(out_dir);
        // 変換済みの点群を指定ディレクトリへ書き出す
        const auto& maps = composer.maps();
        for (size_t i = 0; i < maps.size(); ++i) {
            mc::Cloud tr = mc::MapComposer::transformCloud2D(maps[i].map, T[i]);
            std::string fname = (fs::path(out_dir) / (std::string("map") + std::to_string(i+1) + "_in1.pcd")).string();
            pcl::io::savePCDFileASCII(fname, tr);
        }
    }
    return 0;
}
