#include "map_composer/map_composer.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

using mc::Cloud;
using mc::Point;

/**
 * @brief L字形状の幾何情報を保持する内部用構造体。
 */
struct LFeat {
    Eigen::Vector2d lc;
    Eigen::Vector2d p01;
    Eigen::Vector2d d1;
    Eigen::Vector2d p02;
    Eigen::Vector2d d2;
    double l1;
    double l2;
    double ldeg;
    Eigen::Vector2d e1a;
    Eigen::Vector2d e1b;
    Eigen::Vector2d e2a;
    Eigen::Vector2d e2b;
};

/**
 * @brief 点と直線の距離を返す。
 */
double pointLineDistance(const Eigen::Vector2d& P, const Eigen::Vector2d& p0, const Eigen::Vector2d& dir) {
    Eigen::Vector2d u = P - p0;
    Eigen::Vector2d proj = (u.dot(dir)) * dir;
    return (u - proj).norm();
}

/**
 * @brief 点群からL字特徴を抽出する。
 */
std::vector<LFeat> detectLFeatures(const Cloud& cl) {
    std::vector<LFeat> feats;
    const int n = static_cast<int>(cl.size());
    if (n < 20) {
        return feats;
    }

    Eigen::MatrixXd X(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = cl[i].x;
        X(i, 1) = cl[i].y;
    }

    const double eps = 0.12;
    const int min_samples = 8;
    std::vector<std::vector<int>> neighbors(n);
    for (int i = 0; i < n; ++i) {
        neighbors[i].push_back(i);
        for (int j = i + 1; j < n; ++j) {
            double dx = X(i, 0) - X(j, 0);
            double dy = X(i, 1) - X(j, 1);
            if (std::sqrt(dx * dx + dy * dy) <= eps) {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
            }
        }
    }
    std::vector<int> labels(n, -1);
    std::vector<char> visited(n, 0);
    int cid = 0;
    for (int i = 0; i < n; ++i) {
        if (visited[i]) {
            continue;
        }
        visited[i] = 1;
        const auto& direct = neighbors[i];
        if (static_cast<int>(direct.size()) < min_samples) {
            labels[i] = -1;
            continue;
        }
        for (int v : direct) {
            labels[v] = cid;
        }
        std::vector<int> stack(direct.begin(), direct.end());
        while (!stack.empty()) {
            int j = stack.back();
            stack.pop_back();
            if (visited[j]) {
                continue;
            }
            visited[j] = 1;
            const auto& near_j = neighbors[j];
            if (static_cast<int>(near_j.size()) >= min_samples) {
                for (int k : near_j) {
                    if (labels[k] != cid) {
                        labels[k] = cid;
                        stack.push_back(k);
                    }
                }
            }
        }
        cid++;
    }

    for (int c = 0; c < cid; ++c) {
        std::vector<int> idx;
        idx.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (labels[i] == c) {
                idx.push_back(i);
            }
        }
        if (static_cast<int>(idx.size()) < std::max(20, min_samples * 2)) {
            continue;
        }

        Eigen::MatrixXd M(idx.size(), 2);
        for (size_t k = 0; k < idx.size(); ++k) {
            M.row(k) = X.row(idx[k]);
        }
        Eigen::Vector2d p01 = M.colwise().mean();
        Eigen::MatrixXd Y = M.rowwise() - p01.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y, Eigen::ComputeThinV);
        Eigen::Vector2d d1 = svd.matrixV().col(0);
        d1.normalize();

        auto dist_to = [&](const Eigen::MatrixXd& P, const Eigen::Vector2d& p0, const Eigen::Vector2d& d) {
            Eigen::VectorXd r(P.rows());
            for (Eigen::Index row = 0; row < P.rows(); ++row) {
                Eigen::Vector2d diff = P.row(row).transpose() - p0;
                double proj = diff.dot(d);
                Eigen::Vector2d residual = diff - proj * d;
                r(row) = residual.norm();
            }
            return r;
        };
        Eigen::VectorXd dperp = dist_to(M, p01, d1);

        std::vector<double> tmp(dperp.data(), dperp.data() + dperp.size());
        std::sort(tmp.begin(), tmp.end());
        double m1 = tmp[static_cast<int>(std::floor(0.10 * (tmp.size() - 1)))];
        double m2 = tmp[static_cast<int>(std::floor(0.90 * (tmp.size() - 1)))];
        std::vector<char> assign1(dperp.size(), 0);
        for (int it = 0; it < 10; ++it) {
            int c1 = 0;
            int c2 = 0;
            double s1 = 0.0;
            double s2 = 0.0;
            for (int i = 0; i < dperp.size(); ++i) {
                double a = std::abs(dperp(i) - m1);
                double b = std::abs(dperp(i) - m2);
                if (a <= b) {
                    assign1[i] = 1;
                    s1 += dperp(i);
                    c1++;
                } else {
                    assign1[i] = 0;
                    s2 += dperp(i);
                    c2++;
                }
            }
            if (c1 == 0 || c2 == 0) {
                break;
            }
            m1 = s1 / c1;
            m2 = s2 / c2;
        }

        bool m1_le_m2 = (m1 <= m2);
        std::vector<int> idx1;
        std::vector<int> idx2;
        idx1.reserve(idx.size());
        idx2.reserve(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            bool take1 = (assign1[k] ? m1_le_m2 : !m1_le_m2);
            (take1 ? idx1 : idx2).push_back(idx[k]);
        }
        if (static_cast<int>(idx1.size()) < 5 || static_cast<int>(idx2.size()) < 5) {
            continue;
        }

        Eigen::Vector2d p02 = p01;
        Eigen::Vector2d d2(-d1.y(), d1.x());

        auto fit_line_idx = [&](const std::vector<int>& ids, Eigen::Vector2d& p0, Eigen::Vector2d& d) {
            const int m = static_cast<int>(ids.size());
            Eigen::MatrixXd P(m, 2);
            for (int ii = 0; ii < m; ++ii) {
                P.row(ii) = X.row(ids[ii]);
            }
            p0 = P.colwise().mean();
            Eigen::MatrixXd Z = P.rowwise() - p0.transpose();
            Eigen::JacobiSVD<Eigen::MatrixXd> svd2(Z, Eigen::ComputeThinV);
            d = svd2.matrixV().col(0);
            d.normalize();
        };
        fit_line_idx(idx1, p01, d1);
        fit_line_idx(idx2, p02, d2);

        for (int it = 0; it < 6; ++it) {
            std::vector<int> ni1;
            std::vector<int> ni2;
            ni1.reserve(idx.size());
            ni2.reserve(idx.size());
            for (int id : idx) {
                Eigen::Vector2d P = X.row(id).transpose();
                double da = pointLineDistance(P, p01, d1);
                double db = pointLineDistance(P, p02, d2);
                (da <= db ? ni1 : ni2).push_back(id);
            }
            if (static_cast<int>(ni1.size()) < 5 || static_cast<int>(ni2.size()) < 5) {
                break;
            }
            fit_line_idx(ni1, p01, d1);
            fit_line_idx(ni2, p02, d2);
            idx1.swap(ni1);
            idx2.swap(ni2);
        }

        Eigen::VectorXd tt1(idx1.size());
        Eigen::VectorXd tt2(idx2.size());
        for (int k = 0; k < static_cast<int>(idx1.size()); ++k) {
            tt1(k) = (X.row(idx1[k]).transpose() - p01).dot(d1);
        }
        for (int k = 0; k < static_cast<int>(idx2.size()); ++k) {
            tt2(k) = (X.row(idx2[k]).transpose() - p02).dot(d2);
        }
        double t1min = tt1.minCoeff();
        double t1max = tt1.maxCoeff();
        double t2min = tt2.minCoeff();
        double t2max = tt2.maxCoeff();
        double L1 = t1max - t1min;
        double L2 = t2max - t2min;
        if (L1 < 0.2 || L2 < 0.2) {
            continue;
        }

        Eigen::Matrix2d A2;
        A2 << d1, -d2;
        Eigen::Vector2d rhs = p02 - p01;
        double det = A2.determinant();
        Eigen::Vector2d lc;
        if (std::abs(det) < 1e-8) {
            Eigen::Vector2d n(-d1.y(), d1.x());
            double t = (rhs.dot(n)) / (d2.dot(n) + 1e-12);
            Eigen::Vector2d q1 = p02 + t * d2;
            Eigen::Vector2d q0 = p01 + ((q1 - p01).dot(d1)) * d1;
            lc = 0.5 * (q0 + q1);
        } else {
            Eigen::Vector2d ts = A2.inverse() * rhs;
            lc = p01 + ts(0) * d1;
        }
        double t1lc = (lc - p01).dot(d1);
        double t2lc = (lc - p02).dot(d2);
        auto endpoint_frac = [](double t, double tmin, double tmax) {
            double L = std::max(1e-9, tmax - tmin);
            return std::min(std::abs(t - tmin), std::abs(tmax - t)) / L;
        };
        double e1 = endpoint_frac(t1lc, t1min, t1max);
        double e2 = endpoint_frac(t2lc, t2min, t2max);
        double ang = std::acos(std::min(1.0, std::max(-1.0, std::abs(d1.dot(d2))))) * 180.0 / M_PI;
        if (ang < 20.0 || e1 > 0.5 || e2 > 0.5) {
            continue;
        }
        double l1 = std::max(std::abs(t1max - t1lc), std::abs(t1lc - t1min));
        double l2 = std::max(std::abs(t2max - t2lc), std::abs(t2lc - t2min));
        Eigen::Vector2d e1a = p01 + t1min * d1;
        Eigen::Vector2d e1b = p01 + t1max * d1;
        Eigen::Vector2d e2a = p02 + t2min * d2;
        Eigen::Vector2d e2b = p02 + t2max * d2;
        if (l2 > l1) {
            std::swap(l1, l2);
            std::swap(p01, p02);
            std::swap(d1, d2);
            std::swap(e1a, e2a);
            std::swap(e1b, e2b);
        }
        auto align_sign = [&](Eigen::Vector2d& d, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            Eigen::Vector2d va = a - lc;
            Eigen::Vector2d vb = b - lc;
            Eigen::Vector2d vf = (va.squaredNorm() >= vb.squaredNorm()) ? va : vb;
            if (d.dot(vf) < 0) {
                d = -d;
            }
        };
        align_sign(d1, e1a, e1b);
        align_sign(d2, e2a, e2b);
        feats.push_back({lc, p01, d1, p02, d2, l1, l2, ang, e1a, e1b, e2a, e2b});
    }
    return feats;
}

/**
 * @brief L字特徴の向きから変換を推定する。
 */
Eigen::Isometry2d estimateFromL(const LFeat& s, const LFeat& d) {
    auto angle_from_vec = [](const Eigen::Vector2d& v) {
        return std::atan2(v.y(), v.x()) * 180.0 / M_PI;
    };
    auto normang = [](double a) {
        a = std::fmod(a + 180.0, 360.0);
        if (a < 0) {
            a += 360.0;
        }
        return a - 180.0;
    };
    const double ang_s1 = angle_from_vec(s.d1);
    const double ang_s2 = angle_from_vec(s.d2);
    const double ang_d1 = angle_from_vec(d.d1);
    const double ang_d2 = angle_from_vec(d.d2);

    auto eval_cand = [&](double ang_d_primary, double ang_d_other) {
        double base0 = normang(ang_d_primary - ang_s1);
        auto score_for = [&](double base) {
            double r = std::min(
                std::abs(normang(ang_d_other - (ang_s2 + base))),
                std::abs(normang(ang_d_other - (ang_s2 + base + 180.0))));
            return r;
        };
        double r0 = score_for(base0);
        double base1 = normang(base0 + 180.0);
        double r1 = score_for(base1);
        return (r0 <= r1) ? std::make_pair(r0, base0) : std::make_pair(r1, base1);
    };

    auto c1 = eval_cand(ang_d1, ang_d2);
    auto c2 = eval_cand(ang_d2, ang_d1);
    double th = (c1.first <= c2.first) ? c1.second : c2.second;

    Eigen::Rotation2D<double> Rdeg(th * M_PI / 180.0);
    Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
    iso.linear() = Rdeg.toRotationMatrix();
    iso.translation() = d.lc - iso.linear() * s.lc;
    return iso;
}

/**
 * @brief L字特徴の曖昧性を考慮した変換候補を生成する。
 */
std::vector<Eigen::Isometry2d> generateLCandidates(const LFeat& s, const LFeat& d) {
    auto angle_from_vec = [](const Eigen::Vector2d& v) {
        return std::atan2(v.y(), v.x()) * 180.0 / M_PI;
    };
    auto normang = [](double a) {
        a = std::fmod(a + 180.0, 360.0);
        if (a < 0) {
            a += 360.0;
        }
        return a - 180.0;
    };
    const double ang_s1 = angle_from_vec(s.d1);
    const double ang_d1 = angle_from_vec(d.d1);
    const double ang_d2 = angle_from_vec(d.d2);
    std::vector<double> bases = {normang(ang_d1 - ang_s1), normang(ang_d2 - ang_s1)};

    std::vector<Eigen::Isometry2d> res;
    for (double th : bases) {
        Eigen::Rotation2D<double> Rdeg(th * M_PI / 180.0);
        Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
        iso.linear() = Rdeg.toRotationMatrix();
        iso.translation() = d.lc - iso.linear() * s.lc;
        res.push_back(iso);
    }
    return res;
}

/**
 * @brief 変換後の端点整合度を計算する。
 */
double endpointScore(const LFeat& s, const LFeat& d, const Eigen::Isometry2d& iso) {
    auto tr = [&](const Eigen::Vector2d& p) {
        return iso.linear() * p + iso.translation();
    };
    Eigen::Vector2d se1a = tr(s.e1a);
    Eigen::Vector2d se1b = tr(s.e1b);
    Eigen::Vector2d se2a = tr(s.e2a);
    Eigen::Vector2d se2b = tr(s.e2b);
    auto pair_cost = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c, const Eigen::Vector2d& dpt) {
        double c1 = (a - c).norm() + (b - dpt).norm();
        double c2 = (a - dpt).norm() + (b - c).norm();
        return std::min(c1, c2) * 0.5;
    };
    double opt1 = pair_cost(se1a, se1b, d.e1a, d.e1b) + pair_cost(se2a, se2b, d.e2a, d.e2b);
    double opt2 = pair_cost(se1a, se1b, d.e2a, d.e2b) + pair_cost(se2a, se2b, d.e1a, d.e1b);
    return std::min(opt1, opt2) * 0.5;
}

/**
 * @brief 近傍点のロバストなMSEを計測する。
 */
double pointsAlignmentMSE(const Eigen::Isometry2d& iso,
                          const Cloud& raw1,
                          const Cloud& raw2,
                          const Eigen::Vector2d* center_s = nullptr,
                          const Eigen::Vector2d* center_d = nullptr,
                          double radius = 1.5) {
    std::vector<Eigen::Vector2d> src;
    src.reserve(raw1.size());
    if (center_s) {
        for (const auto& p : raw1) {
            Eigen::Vector2d q(p.x, p.y);
            if ((q - *center_s).norm() <= radius) {
                src.push_back(q);
            }
        }
        if (src.empty()) {
            for (const auto& p : raw1) {
                src.emplace_back(p.x, p.y);
            }
        }
    } else {
        for (const auto& p : raw1) {
            src.emplace_back(p.x, p.y);
        }
    }

    std::vector<Eigen::Vector2d> dst;
    dst.reserve(raw2.size());
    if (center_d) {
        for (const auto& p : raw2) {
            Eigen::Vector2d q(p.x, p.y);
            if ((q - *center_d).norm() <= radius) {
                dst.push_back(q);
            }
        }
        if (dst.empty()) {
            for (const auto& p : raw2) {
                dst.emplace_back(p.x, p.y);
            }
        }
    } else {
        for (const auto& p : raw2) {
            dst.emplace_back(p.x, p.y);
        }
    }
    if (src.empty() || dst.empty()) {
        return std::numeric_limits<double>::max();
    }

    std::vector<double> dists;
    dists.reserve(src.size());
    for (const auto& q : src) {
        Eigen::Vector2d m = iso.linear() * q + iso.translation();
        double best = std::numeric_limits<double>::max();
        for (const auto& r : dst) {
            double d = (r - m).norm();
            if (d < best) {
                best = d;
            }
        }
        dists.push_back(best);
    }
    if (dists.empty()) {
        return std::numeric_limits<double>::max();
    }
    std::sort(dists.begin(), dists.end());
    int k = std::max(1, static_cast<int>(dists.size()) / 2);
    double s = 0.0;
    for (int i = 0; i < k; ++i) {
        s += dists[i];
    }
    return s / k;
}

/**
 * @brief L字特徴集合から最適な変換を推定する。
 */
Eigen::Isometry2d estimateFromLs(const std::vector<LFeat>& F1,
                                 const std::vector<LFeat>& F2,
                                 const Cloud& raw1,
                                 const Cloud& raw2) {
    if (F1.empty() || F2.empty()) {
        return Eigen::Isometry2d::Identity();
    }

    auto count_inliers = [&](const Eigen::Isometry2d& iso, double& mse) {
        const double tol = 0.3;
        int used_ct = 0;
        std::vector<char> used(F2.size(), 0);
        double se = 0.0;
        int cnt = 0;
        for (size_t i = 0; i < F1.size(); ++i) {
            Eigen::Vector2d m = iso.linear() * F1[i].lc + iso.translation();
            int best = -1;
            double bd = std::numeric_limits<double>::max();
            for (size_t j = 0; j < F2.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                double d = (F2[j].lc - m).norm();
                if (d < bd) {
                    bd = d;
                    best = static_cast<int>(j);
                }
            }
            if (best >= 0 && bd <= tol) {
                used[best] = 1;
                used_ct++;
                se += bd * bd;
                cnt++;
            }
        }
        mse = (cnt > 0) ? se / cnt : std::numeric_limits<double>::max();
        return used_ct;
    };

    bool has_best = false;
    int best_inl = 0;
    double best_mse = std::numeric_limits<double>::max();
    double best_pts = std::numeric_limits<double>::max();
    double best_all = std::numeric_limits<double>::max();
    double best_end = std::numeric_limits<double>::max();
    Eigen::Isometry2d best = Eigen::Isometry2d::Identity();
    const bool singleL = (std::min(F1.size(), F2.size()) == 1);

    for (size_t i = 0; i < F1.size(); ++i) {
        for (size_t j = 0; j < F2.size(); ++j) {
            if (std::abs(F1[i].ldeg - F2[j].ldeg) > 10.0) {
                continue;
            }
            if (singleL) {
                auto cands = generateLCandidates(F1[i], F2[j]);
                for (const auto& iso : cands) {
                    double mse = 0.0;
                    int inl = count_inliers(iso, mse);
                    double mse_pts = pointsAlignmentMSE(iso, raw1, raw2, &F1[i].lc, &F2[j].lc, 1.5);
                    double mse_all = pointsAlignmentMSE(iso, raw1, raw2);
                    double e_end = endpointScore(F1[i], F2[j], iso);
                    if (!has_best || (e_end < best_end) ||
                        (std::abs(e_end - best_end) < 1e-6 &&
                         ((mse_all < best_all) ||
                          (std::abs(mse_all - best_all) < 1e-6 &&
                           (mse_pts < best_pts ||
                            (std::abs(mse_pts - best_pts) < 1e-6 &&
                             (inl > best_inl || (inl == best_inl && mse < best_mse)))))))) {
                        has_best = true;
                        best_end = e_end;
                        best_all = mse_all;
                        best_pts = mse_pts;
                        best_inl = inl;
                        best_mse = mse;
                        best = iso;
                    }
                }
            } else {
                Eigen::Isometry2d iso = estimateFromL(F1[i], F2[j]);
                double mse = 0.0;
                int inl = count_inliers(iso, mse);
                double mse_pts = pointsAlignmentMSE(iso, raw1, raw2, &F1[i].lc, &F2[j].lc, 1.5);
                double mse_all = pointsAlignmentMSE(iso, raw1, raw2);
                double e_end = endpointScore(F1[i], F2[j], iso);
                if (!has_best || inl > best_inl ||
                    (inl == best_inl &&
                     (e_end < best_end ||
                      (std::abs(e_end - best_end) < 1e-6 &&
                       (mse_pts < best_pts ||
                        (std::abs(mse_pts - best_pts) < 1e-6 && mse < best_mse)))))) {
                    has_best = true;
                    best_inl = inl;
                    best_mse = mse;
                    best_pts = mse_pts;
                    best_all = mse_all;
                    best_end = e_end;
                    best = iso;
                }
            }
        }
    }
    if (!has_best) {
        best = estimateFromL(F1.front(), F2.front());
    }
    return best;
}

/**
 * @brief 円形ランドマークの中心点を抽出する。
 */
std::vector<Eigen::Vector2d> detectCircleCenters(const Cloud& cl) {
    const int n = static_cast<int>(cl.size());
    if (n == 0) {
        return {};
    }
    std::vector<std::vector<int>> neighbors(n);
    const double eps = 0.12;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = cl[i].x - cl[j].x;
            double dy = cl[i].y - cl[j].y;
            if (std::sqrt(dx * dx + dy * dy) <= eps) {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
            }
        }
    }
    std::vector<int> label(n, -1);
    int cid = 0;
    for (int i = 0; i < n; ++i) {
        if (label[i] != -1) {
            continue;
        }
        std::queue<int> qq;
        qq.push(i);
        label[i] = cid;
        while (!qq.empty()) {
            int u = qq.front();
            qq.pop();
            for (int v : neighbors[u]) {
                if (label[v] == -1) {
                    label[v] = cid;
                    qq.push(v);
                }
            }
        }
        cid++;
    }

    std::vector<Eigen::Vector2d> centers;
    for (int c = 0; c < cid; ++c) {
        std::vector<int> idx;
        for (int i = 0; i < n; ++i) {
            if (label[i] == c) {
                idx.push_back(i);
            }
        }
        if (static_cast<int>(idx.size()) < 12) {
            continue;
        }
        Eigen::MatrixXd A(idx.size(), 3);
        Eigen::VectorXd b(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            double x = cl[idx[k]].x;
            double y = cl[idx[k]].y;
            A(k, 0) = x;
            A(k, 1) = y;
            A(k, 2) = 1.0;
            b(k) = -(x * x + y * y);
        }
        Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(b);
        double A_ = coeffs(0);
        double B_ = coeffs(1);
        double C_ = coeffs(2);
        double cx = -A_ / 2.0;
        double cy = -B_ / 2.0;
        double R2 = (A_ * A_ + B_ * B_) / 4.0 - C_;
        if (R2 <= 0) {
            continue;
        }
        double R = std::sqrt(R2);
        double sum = 0.0;
        for (int id : idx) {
            double dx = cl[id].x - cx;
            double dy = cl[id].y - cy;
            double r = std::sqrt(dx * dx + dy * dy);
            sum += std::abs(r - R);
        }
        double resn = (sum / idx.size()) / std::max(1e-9, R);
        if (R >= 0.015 && R <= 0.05 && resn < 0.08) {
            centers.emplace_back(cx, cy);
        }
    }
    return centers;
}

/**
 * @brief 円中心から変換を推定する (RANSAC + Procrustes)。
 */
Eigen::Isometry2d estimateFromCircleCenters(const std::vector<Eigen::Vector2d>& S,
                                            const std::vector<Eigen::Vector2d>& D) {
    if (S.size() < 2 || D.size() < 2) {
        return Eigen::Isometry2d::Identity();
    }

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> ds(0, static_cast<int>(S.size()) - 1);
    std::uniform_int_distribution<int> dd(0, static_cast<int>(D.size()) - 1);

    auto compute_iso = [](const Eigen::Vector2d& p1,
                          const Eigen::Vector2d& p2,
                          const Eigen::Vector2d& q1,
                          const Eigen::Vector2d& q2) {
        Eigen::Vector2d vs = p2 - p1;
        Eigen::Vector2d vd = q2 - q1;
        double ang = std::atan2(vd.y(), vd.x()) - std::atan2(vs.y(), vs.x());
        Eigen::Rotation2D<double> R(ang);
        Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
        iso.linear() = R.toRotationMatrix();
        iso.translation() = q1 - iso.linear() * p1;
        return iso;
    };

    auto count_inliers = [&](const Eigen::Isometry2d& iso, std::vector<std::pair<int, int>>& pairs) {
        const double tol = 0.15;
        pairs.clear();
        std::vector<char> used(D.size(), 0);
        for (size_t i = 0; i < S.size(); ++i) {
            Eigen::Vector2d m = iso.linear() * S[i] + iso.translation();
            int bestj = -1;
            double bestd = std::numeric_limits<double>::max();
            for (size_t j = 0; j < D.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                double d = (D[j] - m).norm();
                if (d < bestd) {
                    bestd = d;
                    bestj = static_cast<int>(j);
                }
            }
            if (bestj >= 0 && bestd <= tol) {
                used[bestj] = 1;
                pairs.emplace_back(static_cast<int>(i), bestj);
            }
        }
        return static_cast<int>(pairs.size());
    };

    int best_inl = -1;
    Eigen::Isometry2d best_iso = Eigen::Isometry2d::Identity();
    std::vector<std::pair<int, int>> best_pairs;
    for (int it = 0; it < 300; ++it) {
        int i1 = ds(rng);
        int i2 = ds(rng);
        if (i1 == i2) {
            continue;
        }
        int j1 = dd(rng);
        int j2 = dd(rng);
        if (j1 == j2) {
            continue;
        }
        Eigen::Isometry2d iso = compute_iso(S[i1], S[i2], D[j1], D[j2]);
        std::vector<std::pair<int, int>> pairs;
        int ninl = count_inliers(iso, pairs);
        if (ninl > best_inl) {
            best_inl = ninl;
            best_iso = iso;
            best_pairs = pairs;
        }
    }

    if (best_inl >= 2) {
        Eigen::Vector2d cs(0, 0);
        Eigen::Vector2d cd(0, 0);
        for (auto& pr : best_pairs) {
            cs += S[pr.first];
            cd += D[pr.second];
        }
        cs /= static_cast<double>(best_pairs.size());
        cd /= static_cast<double>(best_pairs.size());
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        for (auto& pr : best_pairs) {
            Eigen::Vector2d xs = S[pr.first] - cs;
            Eigen::Vector2d xd = D[pr.second] - cd;
            H += xs * xd.transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
        if (R.determinant() < 0) {
            Eigen::Matrix2d V = svd.matrixV();
            V.col(1) *= -1;
            R = V * svd.matrixU().transpose();
        }
        Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
        iso.linear() = R;
        iso.translation() = cd - R * cs;
        return iso;
    }
    return best_iso;
}

/**
 * @brief L字特徴ペアのインデックスを選択する。
 */
std::pair<int, int> chooseBestLPair(const std::vector<LFeat>& F1,
                                    const std::vector<LFeat>& F2,
                                    const Cloud& raw1,
                                    const Cloud& raw2) {
    int bi = -1;
    int bj = -1;
    bool has = false;
    int best_inl = 0;
    double best_pts = std::numeric_limits<double>::max();

    auto count_inliers = [&](const Eigen::Isometry2d& iso) {
        const double tol = 0.3;
        int used_ct = 0;
        std::vector<char> used(F2.size(), 0);
        for (size_t i = 0; i < F1.size(); ++i) {
            Eigen::Vector2d m = iso.linear() * F1[i].lc + iso.translation();
            int best = -1;
            double bd = std::numeric_limits<double>::max();
            for (size_t j = 0; j < F2.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                double d = (F2[j].lc - m).norm();
                if (d < bd) {
                    bd = d;
                    best = static_cast<int>(j);
                }
            }
            if (best >= 0 && bd <= tol) {
                used[best] = 1;
                used_ct++;
            }
        }
        return used_ct;
    };

    for (size_t i = 0; i < F1.size(); ++i) {
        for (size_t j = 0; j < F2.size(); ++j) {
            if (std::abs(F1[i].ldeg - F2[j].ldeg) > 10.0) {
                continue;
            }
            Eigen::Isometry2d iso = estimateFromL(F1[i], F2[j]);
            int inl = count_inliers(iso);
            double pts = pointsAlignmentMSE(iso, raw1, raw2, &F1[i].lc, &F2[j].lc);
            if (!has || inl > best_inl || (inl == best_inl && pts < best_pts)) {
                has = true;
                best_inl = inl;
                best_pts = pts;
                bi = static_cast<int>(i);
                bj = static_cast<int>(j);
            }
        }
    }
    if (!has) {
        return {0, 0};
    }
    return {bi, bj};
}

/**
 * @brief 点群を変換した際のターゲット集合に対するロバスト誤差。
 */
double robustMSEToTarget(const Eigen::Isometry2d& Ta,
                         const Cloud& A,
                         const std::vector<Eigen::Vector2d>& targetPts) {
    if (A.empty() || targetPts.empty()) {
        return std::numeric_limits<double>::max();
    }
    std::vector<double> dd;
    dd.reserve(A.size());
    for (const auto& p : A) {
        Eigen::Vector2d u2 = Ta.linear() * Eigen::Vector2d(p.x, p.y) + Ta.translation();
        double best = std::numeric_limits<double>::max();
        for (const auto& t : targetPts) {
            double d = (t - u2).norm();
            if (d < best) {
                best = d;
            }
        }
        dd.push_back(best);
    }
    if (dd.empty()) {
        return std::numeric_limits<double>::max();
    }
    std::sort(dd.begin(), dd.end());
    int k = std::max(1, static_cast<int>(dd.size()) / 2);
    double s = 0.0;
    for (int i = 0; i < k; ++i) {
        s += dd[i];
    }
    return s / k;
}

/**
 * @brief 隣接辺を一意に識別するキー。
 */
long long edgeKey(int u, int v) {
    return (static_cast<long long>(u) << 32) ^ static_cast<unsigned long long>(v);
}

} // namespace

namespace mc {

bool MapComposer::load_pcds(const std::vector<std::string>& pcd_paths) {
    m_map_info.clear();
    for (const auto& p : pcd_paths) {
        map_info info;
        if (pcl::io::loadPCDFile<Point>(p, info.map) < 0) {
            std::cerr << "Failed to load PCD: " << p << std::endl;
            return false;
        }
        m_map_info.push_back(std::move(info));
    }
    return true;
}

bool MapComposer::load_adjacency_csv(const std::string& csv_path) {
    std::ifstream ifs(csv_path);
    if (!ifs) {
        std::cerr << "Failed to open adjacency CSV: " << csv_path << std::endl;
        return false;
    }
    std::vector<std::vector<int>> mat;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<int> row;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stoi(cell));
            } catch (...) {
                row.push_back(0);
            }
        }
        if (!row.empty()) {
            mat.push_back(std::move(row));
        }
    }
    if (mat.empty() || mat.size() != mat[0].size()) {
        std::cerr << "Adjacency must be a non-empty square matrix" << std::endl;
        return false;
    }
    const size_t N = mat.size();
    m_adj.assign(N, {});
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            if (mat[i][j] != 0) {
                m_adj[i].push_back(static_cast<int>(j));
            }
        }
    }
    if (m_map_info.size() < N) {
        m_map_info.resize(N);
    }
    for (size_t i = 0; i < N; ++i) {
        m_map_info[i].next_no = m_adj[i];
    }
    return true;
}

std::pair<bool, Eigen::Matrix4f> MapComposer::icp2D(const Cloud::ConstPtr& src,
                                                    const Cloud::ConstPtr& dst,
                                                    double max_corr,
                                                    int max_iter,
                                                    double trans_eps,
                                                    double fitness_eps) {
    pcl::IterativeClosestPoint<Point, Point> icp;
    icp.setMaxCorrespondenceDistance(max_corr);
    icp.setMaximumIterations(max_iter);
    icp.setTransformationEpsilon(trans_eps);
    icp.setEuclideanFitnessEpsilon(fitness_eps);
    icp.setUseReciprocalCorrespondences(true);
    icp.setInputSource(src);
    icp.setInputTarget(dst);
    Cloud aligned;
    Eigen::Vector4f c_src;
    Eigen::Vector4f c_dst;
    pcl::compute3DCentroid(*src, c_src);
    pcl::compute3DCentroid(*dst, c_dst);
    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
    guess(0, 3) = c_dst.x() - c_src.x();
    guess(1, 3) = c_dst.y() - c_src.y();
    icp.align(aligned, guess);
    const bool ok = icp.hasConverged();
    return {ok, icp.getFinalTransformation()};
}

Eigen::Isometry2d MapComposer::toIso2D(const Eigen::Matrix4f& T) {
    Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
    double c = static_cast<double>(T(0, 0));
    double s = static_cast<double>(T(1, 0));
    Eigen::Rotation2D<double> R(std::atan2(s, c));
    iso.linear() = R.toRotationMatrix();
    iso.translation() = Eigen::Vector2d(static_cast<double>(T(0, 3)), static_cast<double>(T(1, 3)));
    return iso;
}

Eigen::Isometry2d MapComposer::composeIso2D(const Eigen::Isometry2d& ab, const Eigen::Isometry2d& bc) {
    Eigen::Isometry2d out = Eigen::Isometry2d::Identity();
    out.linear() = bc.linear() * ab.linear();
    out.translation() = bc.linear() * ab.translation() + bc.translation();
    return out;
}

Eigen::Isometry2d MapComposer::invertIso2D(const Eigen::Isometry2d& iso) {
    Eigen::Isometry2d inv = Eigen::Isometry2d::Identity();
    inv.linear() = iso.linear().transpose();
    inv.translation() = -inv.linear() * iso.translation();
    return inv;
}

Cloud MapComposer::transformCloud2D(const Cloud& in, const Eigen::Isometry2d& iso) {
    Cloud out;
    out.reserve(in.size());
    for (const auto& pt : in) {
        Eigen::Vector2d p(pt.x, pt.y);
        Eigen::Vector2d q = iso.linear() * p + iso.translation();
        Point o;
        o.x = static_cast<float>(q.x());
        o.y = static_cast<float>(q.y());
        o.z = pt.z;
        o.intensity = pt.intensity;
        out.push_back(o);
    }
    return out;
}

void MapComposer::rebuildAdjacencyFromNextNo() {
    const size_t N = m_map_info.size();
    m_adj.assign(N, {});
    for (size_t i = 0; i < N; ++i) {
        const auto& next = m_map_info[i].next_no;
        auto& adj_list = m_adj[i];
        adj_list.reserve(next.size());
        std::unordered_set<int> seen;
        for (int dst : next) {
            if (dst < 0 || static_cast<size_t>(dst) >= N) {
                continue;
            }
            if (!seen.insert(dst).second) {
                continue;
            }
            adj_list.push_back(dst);
        }
    }
}

bool MapComposer::composeMaps() {
    const size_t N = m_map_info.size();
    if (N == 0) {
        return false;
    }
    if (m_adj.empty()) {
        m_transform.assign(N, Eigen::Isometry2d::Identity());
        return true;
    }

    std::vector<std::vector<Eigen::Vector2d>> circle_centers(N);
    std::vector<std::vector<LFeat>> Lfeats(N);
    for (size_t i = 0; i < N; ++i) {
        circle_centers[i] = detectCircleCenters(m_map_info[i].map);
        Lfeats[i] = detectLFeatures(m_map_info[i].map);
    }

    std::unordered_map<long long, std::vector<Eigen::Isometry2d>> edge_vu_cands;
    std::unordered_map<long long, Eigen::Isometry2d> edge_uv;

    for (size_t u = 0; u < N; ++u) {
        for (int v : m_adj[u]) {
            long long key_uvv = edgeKey(static_cast<int>(u), v);
            if (edge_uv.find(key_uvv) != edge_uv.end()) {
                continue;
            }
            Eigen::Isometry2d uv = Eigen::Isometry2d::Identity();
            if (circle_centers[u].size() >= 2 && circle_centers[v].size() >= 2) {
                uv = estimateFromCircleCenters(circle_centers[u], circle_centers[v]);
                edge_vu_cands[edgeKey(v, static_cast<int>(u))] = {invertIso2D(uv)};
            } else if (!Lfeats[u].empty() && !Lfeats[v].empty()) {
                if (std::min(Lfeats[u].size(), Lfeats[v].size()) == 1) {
                    auto pair_idx = chooseBestLPair(Lfeats[u], Lfeats[v], m_map_info[u].map, m_map_info[v].map);
                    auto cands_uv = generateLCandidates(Lfeats[u][pair_idx.first], Lfeats[v][pair_idx.second]);
                    if (!cands_uv.empty()) {
                        size_t best_idx = 0;
                        double best_score = std::numeric_limits<double>::max();
                        for (size_t k = 0; k < cands_uv.size(); ++k) {
                            double score = pointsAlignmentMSE(cands_uv[k], m_map_info[u].map, m_map_info[v].map,
                                                              &Lfeats[u][pair_idx.first].lc,
                                                              &Lfeats[v][pair_idx.second].lc, 1.5);
                            if (score < best_score) {
                                best_score = score;
                                best_idx = k;
                            }
                        }
                        uv = cands_uv[best_idx];
                        std::vector<Eigen::Isometry2d> vu_cands;
                        vu_cands.reserve(cands_uv.size());
                        for (const auto& cand : cands_uv) {
                            vu_cands.push_back(invertIso2D(cand));
                        }
                        edge_vu_cands[edgeKey(v, static_cast<int>(u))] = std::move(vu_cands);
                    } else {
                        uv = estimateFromL(Lfeats[u][pair_idx.first], Lfeats[v][pair_idx.second]);
                        edge_vu_cands[edgeKey(v, static_cast<int>(u))] = {invertIso2D(uv)};
                    }
                } else {
                    uv = estimateFromLs(Lfeats[u], Lfeats[v], m_map_info[u].map, m_map_info[v].map);
                    edge_vu_cands[edgeKey(v, static_cast<int>(u))] = {invertIso2D(uv)};
                }
            } else {
                Cloud::ConstPtr src(new Cloud(m_map_info[u].map));
                Cloud::ConstPtr dst(new Cloud(m_map_info[v].map));
                auto [ok, T] = icp2D(src, dst, 2.0, 100, 1e-8, 1e-8);
                (void)ok;
                uv = toIso2D(T);
                edge_vu_cands[edgeKey(v, static_cast<int>(u))] = {invertIso2D(uv)};
            }
            edge_uv[key_uvv] = uv;
            edge_uv[edgeKey(v, static_cast<int>(u))] = invertIso2D(uv);
        }
    }

    m_transform.assign(N, Eigen::Isometry2d::Identity());
    std::vector<char> visited(N, 0);
    std::queue<int> q;
    visited[0] = 1;
    q.push(0);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : m_adj[u]) {
            if (visited[v]) {
                continue;
            }
            auto cand_it = edge_vu_cands.find(edgeKey(v, u));
            Eigen::Isometry2d best_v1 = Eigen::Isometry2d::Identity();
            bool chosen = false;
            if (cand_it != edge_vu_cands.end()) {
                std::vector<Eigen::Vector2d> targetPts;
                targetPts.reserve(1024);
                for (size_t k = 0; k < N; ++k) {
                    if (!visited[k]) {
                        continue;
                    }
                    for (const auto& p : m_map_info[k].map) {
                        Eigen::Vector2d qpt(p.x, p.y);
                        Eigen::Vector2d tpt = m_transform[k].linear() * qpt + m_transform[k].translation();
                        targetPts.push_back(tpt);
                    }
                }
                double best_err = std::numeric_limits<double>::max();
                for (const auto& vu : cand_it->second) {
                    Eigen::Isometry2d v1 = composeIso2D(vu, m_transform[u]);
                    double e = robustMSEToTarget(v1, m_map_info[v].map, targetPts);
                    if (!chosen || e < best_err) {
                        chosen = true;
                        best_err = e;
                        best_v1 = v1;
                    }
                }
            }
            if (!chosen) {
                auto uv_it = edge_uv.find(edgeKey(v, u));
                if (uv_it != edge_uv.end()) {
                    best_v1 = composeIso2D(uv_it->second, m_transform[u]);
                }
            }
            m_transform[v] = best_v1;
            visited[v] = 1;
            q.push(v);
        }
    }

    for (size_t i = 1; i < N; ++i) {
        std::vector<Eigen::Vector2d> target;
        target.reserve(1024);
        for (size_t k = 0; k < N; ++k) {
            if (k == i || !visited[k]) {
                continue;
            }
            for (const auto& p : m_map_info[k].map) {
                Eigen::Vector2d qpt(p.x, p.y);
                Eigen::Vector2d tpt = m_transform[k].linear() * qpt + m_transform[k].translation();
                target.push_back(tpt);
            }
        }
        if (target.empty()) {
            continue;
        }
        Eigen::Vector2d c_src(0, 0);
        if (!m_map_info[i].map.empty()) {
            for (const auto& p : m_map_info[i].map) {
                c_src.x() += p.x;
                c_src.y() += p.y;
            }
            c_src /= static_cast<double>(m_map_info[i].map.size());
        }
        Eigen::Vector2d c_map = m_transform[i].linear() * c_src + m_transform[i].translation();
        Eigen::Rotation2D<double> R180(M_PI);
        Eigen::Isometry2d Tflip = Eigen::Isometry2d::Identity();
        Tflip.linear() = R180.toRotationMatrix() * m_transform[i].linear();
        Tflip.translation() = -m_transform[i].translation() + 2.0 * c_map;
        double e_keep = robustMSEToTarget(m_transform[i], m_map_info[i].map, target);
        double e_flip = robustMSEToTarget(Tflip, m_map_info[i].map, target);
        if (e_flip + 1e-9 < e_keep) {
            m_transform[i] = Tflip;
        }
    }
    return true;
}

} // namespace mc
