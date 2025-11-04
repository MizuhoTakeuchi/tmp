#include "map_composer/map_composer.hpp"

#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

namespace fs = std::filesystem;

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --pcd-dir <dir> [--adj <path>] [--write-out <dir>]" << std::endl;
}

static std::vector<std::string> default_pcds(const std::string& dir) {
    std::vector<std::string> out;
    for (int i = 1; i <= 3; ++i) {
        fs::path p = fs::path(dir) / (std::string("map") + std::to_string(i) + ".pcd");
        out.push_back(p.string());
    }
    return out;
}

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

    // Debug: print simple circle stats per map
    auto circle_stats = [](const mc::Cloud& cl, double& R, double& resn) {
        const int n = static_cast<int>(cl.size());
        double mx = 0.0, my = 0.0; for (const auto& p : cl) { mx += p.x; my += p.y; }
        mx /= n; my /= n; std::vector<double> radii(n);
        for (int i = 0; i < n; ++i) { double dx = cl[i].x - mx, dy = cl[i].y - my; radii[i] = std::sqrt(dx*dx + dy*dy); }
        double mean_r = 0.0; for (double r : radii) mean_r += r; mean_r /= n; double var = 0.0; for (double r : radii) { double d = r - mean_r; var += d*d; } var /= std::max(1, n-1); R = mean_r; resn = std::sqrt(var) / std::max(1e-9, R);
    };
    const auto& maps_for_stats = composer.maps();
    for (size_t i = 0; i < maps_for_stats.size(); ++i) {
        double R=0.0, resn=0.0; circle_stats(maps_for_stats[i].map, R, resn);
        std::cout << "circle_stats map" << (i+1) << ": R=" << R << ", resn=" << resn << std::endl;
    }

    bool ok = composer.composeMaps();
    std::cout << (ok ? "composeMaps converged on all edges" : "composeMaps had non-converged edges") << std::endl;

    const auto& T = composer.transforms();
    for (size_t i = 0; i < T.size(); ++i) {
        print_iso_summary(T[i], static_cast<int>(i));
    }

    if (!out_dir.empty()) {
        fs::create_directories(out_dir);
        // write transformed clouds into out_dir
        const auto& maps = composer.maps();
        for (size_t i = 0; i < maps.size(); ++i) {
            mc::Cloud tr = mc::MapComposer::transformCloud2D(maps[i].map, T[i]);
            std::string fname = (fs::path(out_dir) / (std::string("map") + std::to_string(i+1) + "_in1.pcd")).string();
            pcl::io::savePCDFileASCII(fname, tr);
        }
    }
    return 0;
}
