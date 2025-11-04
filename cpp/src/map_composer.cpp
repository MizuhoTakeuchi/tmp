#include "map_composer/map_composer.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <numeric>
#include <random>

namespace mc {

bool MapComposer::load_pcds(const std::vector<std::string>& pcd_paths) {
    m_map_info.clear();
    for (const auto& p : pcd_paths) {
        map_info info;
        if (pcl::io::loadPCDFile<Point>(p, info.map) < 0) {
            std::cerr << "Failed to load PCD: " << p << std::endl;
            return false;
        }
        // ensure intensity exists; if missing, it's fine, default to 0
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
        if (line.empty()) continue;
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
        if (!row.empty()) mat.push_back(std::move(row));
    }
    if (mat.empty() || mat.size() != mat[0].size()) {
        std::cerr << "Adjacency must be a non-empty square matrix" << std::endl;
        return false;
    }
    const size_t N = mat.size();
    m_adj.assign(N, {});
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (mat[i][j] != 0) m_adj[i].push_back(static_cast<int>(j));
        }
    }
    // fill map_info.next_no
    if (m_map_info.size() < N) m_map_info.resize(N);
    for (size_t i = 0; i < N; ++i) {
        m_map_info[i].next_no = m_adj[i];
    }
    return true;
}

std::pair<bool, Eigen::Matrix4f> MapComposer::icp2D(const Cloud::ConstPtr& src, const Cloud::ConstPtr& dst,
                                                    double max_corr, int max_iter, double trans_eps, double fitness_eps) {
    pcl::IterativeClosestPoint<Point, Point> icp;
    icp.setMaxCorrespondenceDistance(max_corr);
    icp.setMaximumIterations(max_iter);
    icp.setTransformationEpsilon(trans_eps);
    icp.setEuclideanFitnessEpsilon(fitness_eps);
    icp.setUseReciprocalCorrespondences(true);
    icp.setInputSource(src);
    icp.setInputTarget(dst);
    Cloud aligned;
    // initial guess: align centroids (translation only)
    Eigen::Vector4f c_src, c_dst;
    pcl::compute3DCentroid(*src, c_src);
    pcl::compute3DCentroid(*dst, c_dst);
    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
    guess(0,3) = c_dst.x() - c_src.x();
    guess(1,3) = c_dst.y() - c_src.y();
    icp.align(aligned, guess);
    const bool ok = icp.hasConverged();
    return {ok, icp.getFinalTransformation()};
}

Eigen::Isometry2d MapComposer::toIso2D(const Eigen::Matrix4f& T) {
    Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
    // rotation in XY plane
    double c = static_cast<double>(T(0, 0));
    double s = static_cast<double>(T(1, 0));
    Eigen::Rotation2D<double> R(std::atan2(s, c));
    iso.linear() = R.toRotationMatrix();
    iso.translation() = Eigen::Vector2d(static_cast<double>(T(0, 3)), static_cast<double>(T(1, 3)));
    return iso;
}

static Eigen::Matrix4f iso2d_to_mat4f(const Eigen::Isometry2d& iso) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(0,0) = static_cast<float>(iso.linear()(0,0));
    T(0,1) = static_cast<float>(iso.linear()(0,1));
    T(1,0) = static_cast<float>(iso.linear()(1,0));
    T(1,1) = static_cast<float>(iso.linear()(1,1));
    T(0,3) = static_cast<float>(iso.translation().x());
    T(1,3) = static_cast<float>(iso.translation().y());
    return T;
}

Eigen::Isometry2d MapComposer::composeIso2D(const Eigen::Isometry2d& ab, const Eigen::Isometry2d& bc) {
    // ab then bc => a->c
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

mc::Cloud MapComposer::transformCloud2D(const Cloud& in, const Eigen::Isometry2d& iso) {
    Cloud out;
    out.reserve(in.size());
    for (const auto& pt : in) {
        Eigen::Vector2d p(pt.x, pt.y);
        Eigen::Vector2d q = iso.linear() * p + iso.translation();
        Point o;
        o.x = static_cast<float>(q.x());
        o.y = static_cast<float>(q.y());
        o.z = pt.z; // keep z as is
        o.intensity = pt.intensity;
        out.push_back(o);
    }
    return out;
}

bool MapComposer::composeMaps() {
    const size_t N = m_map_info.size();
    if (N == 0) return false;
    if (m_adj.empty()) {
        m_transform.assign(N, Eigen::Isometry2d::Identity());
        return true;
    }

    // Helpers: robust circle check via radial std against centroid
    auto circle_stats = [](const Cloud& cl, Eigen::Vector2d& c, double& R, double& resn) -> bool {
        const int n = static_cast<int>(cl.size());
        if (n < 3) return false;
        double mx = 0.0, my = 0.0;
        for (const auto& p : cl) { mx += p.x; my += p.y; }
        mx /= n; my /= n; c = Eigen::Vector2d(mx, my);
        std::vector<double> radii(n);
        for (int i = 0; i < n; ++i) {
            double dx = cl[i].x - mx, dy = cl[i].y - my;
            radii[i] = std::sqrt(dx*dx + dy*dy);
        }
        double mean_r = std::accumulate(radii.begin(), radii.end(), 0.0) / n;
        double var = 0.0; for (double r : radii) { double d = r - mean_r; var += d*d; }
        var /= std::max(1, n-1);
        double std_r = std::sqrt(var);
        R = mean_r;
        resn = std_r / std::max(1e-9, R);
        return true;
    };

    struct LFeat {
        Eigen::Vector2d lc;
        Eigen::Vector2d p01, d1;
        Eigen::Vector2d p02, d2;
        double l1, l2, ldeg;
        Eigen::Vector2d e1a, e1b, e2a, e2b; // endpoints for both lines
    };
    auto eps_clusters = [](const Cloud& cl, double eps, int min_samples) {
        const int n = static_cast<int>(cl.size());
        std::vector<std::vector<int>> neigh(n);
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                double dx = cl[i].x - cl[j].x, dy = cl[i].y - cl[j].y;
                if (std::sqrt(dx*dx + dy*dy) <= eps) { neigh[i].push_back(j); neigh[j].push_back(i); }
            }
        }
        std::vector<int> label(n, -1);
        int cid = 0;
        std::queue<int> qq;
        for (int i = 0; i < n; ++i) {
            if (label[i] != -1) continue;
            // require core-like condition to start
            if ((int)neigh[i].size() < min_samples) { label[i] = -2; continue; }
            label[i] = cid; qq.push(i);
            while (!qq.empty()) { int u = qq.front(); qq.pop(); for (int v : neigh[u]) if (label[v] == -1 || label[v] == -2) { label[v] = cid; qq.push(v);} }
            cid++;
        }
        std::vector<std::vector<int>> clusters(cid);
        for (int i = 0; i < n; ++i) if (label[i] >= 0) clusters[label[i]].push_back(i);
        // filter small clusters
        std::vector<std::vector<int>> out; out.reserve(clusters.size());
        for (auto& c : clusters) if ((int)c.size() >= std::max(min_samples*2, 12)) out.push_back(std::move(c));
        return out;
    };
    auto fit_line = [](const Eigen::MatrixXd& P, const std::vector<int>& idx, Eigen::Vector2d& p0, Eigen::Vector2d& d, Eigen::VectorXd& t){
        const int m = (int)idx.size(); Eigen::MatrixXd M(m,2); for (int k=0;k<m;++k) M.row(k)=P.row(idx[k]);
        p0 = M.colwise().mean(); Eigen::MatrixXd Z = M.rowwise()-p0.transpose(); Eigen::JacobiSVD<Eigen::MatrixXd> svd(Z, Eigen::ComputeThinV);
        d = svd.matrixV().col(0); d.normalize(); t.resize(m); for (int k=0;k<m;++k) t(k)=(M.row(k).transpose()-p0).dot(d);
    };
    auto point_line_dist = [](const Eigen::Vector2d& P, const Eigen::Vector2d& p0, const Eigen::Vector2d& d){ Eigen::Vector2d u=P-p0; Eigen::Vector2d proj=(u.dot(d))*d; return (u-proj).norm(); };
    auto extract_L_features = [&](const Cloud& cl) -> std::vector<LFeat> {
        std::vector<LFeat> feats;
        const int n = (int)cl.size();
        if (n < 20) return feats;
        // Build coordinates and distance matrix (O(n^2))
        Eigen::MatrixXd X(n,2);
        for (int i=0;i<n;++i){ X(i,0)=cl[i].x; X(i,1)=cl[i].y; }
        Eigen::MatrixXd D(n,n);
        for (int i=0;i<n;++i){
            for (int j=0;j<n;++j){ double dx=X(i,0)-X(j,0), dy=X(i,1)-X(j,1); D(i,j)=std::sqrt(dx*dx+dy*dy); }
        }
        const double eps=0.12; const int min_samples=8;
        std::vector<int> labels(n, -1); std::vector<char> visited(n, 0); int cid=0;
        for (int i=0;i<n;++i){
            if (visited[i]) continue; visited[i]=1;
            std::vector<int> neighbors; neighbors.reserve(n);
            for (int j=0;j<n;++j) if (D(i,j) <= eps) neighbors.push_back(j);
            if ((int)neighbors.size() < min_samples) { labels[i] = -1; continue; }
            for (int v: neighbors) labels[v] = cid;
            std::vector<int> stack = neighbors;
            while (!stack.empty()){
                int j = stack.back(); stack.pop_back();
                if (visited[j]) continue; visited[j]=1;
                std::vector<int> n2; n2.reserve(n);
                for (int k=0;k<n;++k) if (D(j,k) <= eps) n2.push_back(k);
                if ((int)n2.size() >= min_samples){
                    for (int k: n2){ if (labels[k] != cid){ labels[k] = cid; stack.push_back(k);} }
                }
            }
            cid++;
        }
        // collect clusters 0..cid-1
        for (int c=0;c<cid;++c){
            std::vector<int> idx; idx.reserve(n);
            for (int i=0;i<n;++i) if (labels[i]==c) idx.push_back(i);
            if ((int)idx.size() < std::max(20, min_samples*2)) continue;
            // initial line1 via PCA
            Eigen::MatrixXd M(idx.size(),2); for (size_t k=0;k<idx.size();++k) M.row(k) = X.row(idx[k]);
            Eigen::Vector2d p01 = M.colwise().mean(); Eigen::MatrixXd Y = M.rowwise() - p01.transpose();
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Y, Eigen::ComputeThinV);
            Eigen::Vector2d d1 = svd.matrixV().col(0); d1.normalize();
            // distances to line1
            auto dist_to = [&](const Eigen::MatrixXd& P, const Eigen::Vector2d& p0, const Eigen::Vector2d& d){
                Eigen::MatrixXd U = P.rowwise() - p0.transpose(); Eigen::VectorXd t = U * d; Eigen::MatrixXd proj = t * d.transpose();
                Eigen::MatrixXd R = U - proj; Eigen::VectorXd r = R.rowwise().norm(); return r;
            };
            Eigen::VectorXd dperp = dist_to(M, p01, d1);
            // k-means on dperp scalar: initialize by 10/90 percentile
            std::vector<double> tmp(dperp.data(), dperp.data()+dperp.size()); std::sort(tmp.begin(), tmp.end());
            double m1 = tmp[(int)std::floor(0.10 * (tmp.size()-1))];
            double m2 = tmp[(int)std::floor(0.90 * (tmp.size()-1))];
            std::vector<char> assign1(dperp.size(), 0);
            for (int it=0; it<10; ++it){
                // assign to nearest mean
                int c1=0, c2=0; double s1=0.0, s2=0.0;
                for (int i=0;i<dperp.size();++i){ double a = std::abs(dperp(i)-m1), b=std::abs(dperp(i)-m2); if (a<=b){ assign1[i]=1; s1+=dperp(i); c1++; } else { assign1[i]=0; s2+=dperp(i); c2++; } }
                if (c1==0 || c2==0) break;
                m1 = s1 / c1; m2 = s2 / c2;
            }
            // smaller-mean cluster -> line1; other -> line2
            bool m1_le_m2 = (m1 <= m2);
            std::vector<int> idx1, idx2; idx1.reserve(idx.size()); idx2.reserve(idx.size());
            for (size_t k=0;k<idx.size();++k){ bool take1 = (assign1[k] ? m1_le_m2 : !m1_le_m2); (take1 ? idx1 : idx2).push_back(idx[k]); }
            if ((int)idx1.size() < 5 || (int)idx2.size() < 5) continue;
            // initialize second line perpendicular to first at same center
            Eigen::Vector2d p02 = p01; Eigen::Vector2d d2(-d1.y(), d1.x());
            Eigen::VectorXd t1, t2;
            // refine by assigning to nearest of two lines and refit, repeat
            auto fit_line_idx = [&](const std::vector<int>& ids, Eigen::Vector2d& p0, Eigen::Vector2d& d, Eigen::VectorXd& t){
                const int m = (int)ids.size(); Eigen::MatrixXd P(m,2); for (int ii=0; ii<m; ++ii) P.row(ii) = X.row(ids[ii]);
                p0 = P.colwise().mean(); Eigen::MatrixXd Z = P.rowwise() - p0.transpose(); Eigen::JacobiSVD<Eigen::MatrixXd> svd2(Z, Eigen::ComputeThinV);
                d = svd2.matrixV().col(0); d.normalize(); t.resize(m); for (int ii=0; ii<m; ++ii) t(ii) = (P.row(ii).transpose()-p0).dot(d);
            };
            fit_line_idx(idx1, p01, d1, t1); fit_line_idx(idx2, p02, d2, t2);
            for (int it=0; it<6; ++it){
                std::vector<int> ni1, ni2; ni1.reserve(idx.size()); ni2.reserve(idx.size());
                for (int id : idx){ Eigen::Vector2d P = X.row(id).transpose(); double da = point_line_dist(P,p01,d1), db = point_line_dist(P,p02,d2); (da<=db ? ni1 : ni2).push_back(id); }
                if ((int)ni1.size()<5 || (int)ni2.size()<5) break;
                fit_line_idx(ni1, p01, d1, t1); fit_line_idx(ni2, p02, d2, t2);
                idx1.swap(ni1); idx2.swap(ni2);
            }
            // extents and intersection
            Eigen::VectorXd tt1(idx1.size()); for (int k=0;k<(int)idx1.size();++k) tt1(k) = (X.row(idx1[k]).transpose()-p01).dot(d1);
            Eigen::VectorXd tt2(idx2.size()); for (int k=0;k<(int)idx2.size();++k) tt2(k) = (X.row(idx2[k]).transpose()-p02).dot(d2);
            double t1min=tt1.minCoeff(), t1max=tt1.maxCoeff(); double t2min=tt2.minCoeff(), t2max=tt2.maxCoeff();
            double L1 = t1max - t1min, L2 = t2max - t2min; if (L1 < 0.2 || L2 < 0.2) continue;
            // line intersection
            Eigen::Matrix2d A2; A2 << d1, -d2; Eigen::Vector2d rhs = p02 - p01; double det = A2.determinant(); Eigen::Vector2d lc;
            if (std::abs(det) < 1e-8){ Eigen::Vector2d n(-d1.y(), d1.x()); double t=(rhs.dot(n))/(d2.dot(n)+1e-12); Eigen::Vector2d q1=p02+t*d2; Eigen::Vector2d q0=p01+((q1-p01).dot(d1))*d1; lc=0.5*(q0+q1);} else { Eigen::Vector2d ts = A2.inverse()*rhs; lc = p01 + ts(0)*d1; }
            double t1lc = (lc - p01).dot(d1), t2lc = (lc - p02).dot(d2);
            auto endpoint_frac = [](double t, double tmin, double tmax){ double L = std::max(1e-9, tmax-tmin); return std::min(std::abs(t-tmin), std::abs(tmax-t)) / L; };
            double e1 = endpoint_frac(t1lc, t1min, t1max), e2 = endpoint_frac(t2lc, t2min, t2max);
            double ang = std::acos(std::min(1.0, std::max(-1.0, std::abs(d1.dot(d2))))) * 180.0 / M_PI;
            if (ang < 20.0 || e1 > 0.5 || e2 > 0.5) continue;
            double l1 = std::max(std::abs(t1max - t1lc), std::abs(t1lc - t1min));
            double l2 = std::max(std::abs(t2max - t2lc), std::abs(t2lc - t2min));
            // compute endpoints for both lines
            Eigen::Vector2d e1a = p01 + t1min * d1; Eigen::Vector2d e1b = p01 + t1max * d1;
            Eigen::Vector2d e2a = p02 + t2min * d2; Eigen::Vector2d e2b = p02 + t2max * d2;
            if (l2 > l1) { std::swap(l1,l2); std::swap(p01,p02); std::swap(d1,d2); std::swap(e1a,e2a); std::swap(e1b,e2b); }
            // sign normalization: d points from lc toward farther endpoint
            auto align_sign = [&](Eigen::Vector2d& d, const Eigen::Vector2d& a, const Eigen::Vector2d& b){
                Eigen::Vector2d va = a - lc, vb = b - lc; Eigen::Vector2d vf = (va.squaredNorm() >= vb.squaredNorm()) ? va : vb; if (d.dot(vf) < 0) d = -d; };
            align_sign(d1, e1a, e1b); align_sign(d2, e2a, e2b);
            feats.push_back({lc,p01,d1,p02,d2,l1,l2,ang,e1a,e1b,e2a,e2b});
        }
        return feats;
    };

    auto estimate_from_L = [](const LFeat& s, const LFeat& d) -> Eigen::Isometry2d {
        auto angle_from_vec = [](const Eigen::Vector2d& v) { return std::atan2(v.y(), v.x()) * 180.0 / M_PI; };
        auto normang = [](double a){ a = std::fmod(a + 180.0, 360.0); if (a < 0) a += 360.0; return a - 180.0; };
        const double ang_s1 = angle_from_vec(s.d1);
        const double ang_s2 = angle_from_vec(s.d2);
        const double ang_d1 = angle_from_vec(d.d1);
        const double ang_d2 = angle_from_vec(d.d2);

        auto eval_cand = [&](double ang_d_primary, double ang_d_other){
            // consider base and base+180 due to line direction ambiguity
            double base0 = normang(ang_d_primary - ang_s1);
            auto score_for = [&](double base){
                double r = std::min(std::abs(normang(ang_d_other - (ang_s2 + base))), std::abs(normang(ang_d_other - (ang_s2 + base + 180.0))));
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
    };

    auto estimate_from_L_candidates = [](const LFeat& s, const LFeat& d) -> std::vector<Eigen::Isometry2d> {
        auto angle_from_vec = [](const Eigen::Vector2d& v) { return std::atan2(v.y(), v.x()) * 180.0 / M_PI; };
        auto normang = [](double a){ a = std::fmod(a + 180.0, 360.0); if (a < 0) a += 360.0; return a - 180.0; };
        const double ang_s1 = angle_from_vec(s.d1);
        const double ang_d1 = angle_from_vec(d.d1);
        const double ang_d2 = angle_from_vec(d.d2);
        std::vector<double> bases = { normang(ang_d1 - ang_s1), normang(ang_d2 - ang_s1) };
        std::vector<Eigen::Isometry2d> res;
        for (double th : bases) {
            Eigen::Rotation2D<double> Rdeg(th * M_PI / 180.0);
            Eigen::Isometry2d iso = Eigen::Isometry2d::Identity();
            iso.linear() = Rdeg.toRotationMatrix();
            iso.translation() = d.lc - iso.linear() * s.lc;
            res.push_back(iso);
        }
        return res;
    };

    // endpoint consistency score between transformed source L and dest L
    auto endpoint_score = [](const LFeat& s, const LFeat& d, const Eigen::Isometry2d& iso){
        auto tr = [&](const Eigen::Vector2d& p){ return iso.linear()*p + iso.translation(); };
        Eigen::Vector2d se1a = tr(s.e1a), se1b = tr(s.e1b), se2a = tr(s.e2a), se2b = tr(s.e2b);
        auto pair_cost = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c, const Eigen::Vector2d& d){
            double c1 = (a-c).norm() + (b-d).norm();
            double c2 = (a-d).norm() + (b-c).norm();
            return std::min(c1, c2) * 0.5; // average per endpoint
        };
        // two mapping options: (line1->line1, line2->line2) or cross
        double opt1 = pair_cost(se1a,se1b, d.e1a,d.e1b) + pair_cost(se2a,se2b, d.e2a,d.e2b);
        double opt2 = pair_cost(se1a,se1b, d.e2a,d.e2b) + pair_cost(se2a,se2b, d.e1a,d.e1b);
        return std::min(opt1, opt2) * 0.5; // average of two lines
    };

    auto points_alignment_mse = [](const Eigen::Isometry2d& iso, const Cloud& raw1, const Cloud& raw2, const Eigen::Vector2d* center_s=nullptr, const Eigen::Vector2d* center_d=nullptr, double radius=1.5) -> double {
        std::vector<Eigen::Vector2d> src;
        src.reserve(raw1.size());
        if (center_s) {
            for (const auto& p : raw1) { Eigen::Vector2d q(p.x,p.y); if ((q - *center_s).norm() <= radius) src.push_back(q); }
            if (src.empty()) { for (const auto& p : raw1) src.emplace_back(p.x,p.y); }
        } else { for (const auto& p : raw1) src.emplace_back(p.x,p.y); }
        std::vector<Eigen::Vector2d> dst;
        dst.reserve(raw2.size());
        if (center_d) {
            for (const auto& p : raw2) { Eigen::Vector2d q(p.x,p.y); if ((q - *center_d).norm() <= radius) dst.push_back(q); }
            if (dst.empty()) { for (const auto& p : raw2) dst.emplace_back(p.x,p.y); }
        } else { for (const auto& p : raw2) dst.emplace_back(p.x,p.y); }
        if (src.empty() || dst.empty()) return 1e9;
        std::vector<double> dists; dists.reserve(src.size());
        for (const auto& q : src) {
            Eigen::Vector2d m = iso.linear()*q + iso.translation();
            double best = 1e9; for (const auto& r : dst) { double d = (r - m).norm(); if (d < best) best = d; }
            dists.push_back(best);
        }
        if (dists.empty()) return 1e9;
        std::sort(dists.begin(), dists.end());
        int k = std::max(1, (int)dists.size()/2);
        double s = 0.0; for (int i=0;i<k;++i) s += dists[i];
        return s / k;
    };

    auto estimate_from_Ls = [&](const std::vector<LFeat>& F1, const std::vector<LFeat>& F2, const Cloud& raw1, const Cloud& raw2) -> Eigen::Isometry2d {
        if (F1.empty() || F2.empty()) return Eigen::Isometry2d::Identity();
        // helper: count inliers by nearest neighbor on L corner points
        auto count_inliers = [&](const Eigen::Isometry2d& iso, double& mse){
            const double tol = 0.3; int used_ct=0; std::vector<char> used(F2.size(),0);
            double se=0.0; int cnt=0;
            for (size_t i=0;i<F1.size();++i){ Eigen::Vector2d m = iso.linear()*F1[i].lc + iso.translation(); int best=-1; double bd=1e9; for (size_t j=0;j<F2.size();++j){ if (used[j]) continue; double d=(F2[j].lc - m).norm(); if (d<bd){ bd=d; best=(int)j; } } if (best>=0 && bd<=tol){ used[best]=1; used_ct++; se+=bd*bd; cnt++; } }
            mse = (cnt>0)? se/cnt : 1e9; return used_ct; };
        // point alignment mse around corners (robust)
        auto points_alignment_mse = [&](const Eigen::Isometry2d& iso, const Eigen::Vector2d& center_s, const Eigen::Vector2d& center_d, double radius=1.5){
            std::vector<Eigen::Vector2d> s; s.reserve(raw1.size()); for (const auto& p: raw1){ Eigen::Vector2d q(p.x,p.y); if ((q-center_s).norm()<=radius) s.push_back(q);} if (s.empty()) { for (const auto& p: raw1) s.emplace_back(p.x,p.y);} 
            std::vector<Eigen::Vector2d> d; d.reserve(raw2.size()); for (const auto& p: raw2){ Eigen::Vector2d q(p.x,p.y); if ((q-center_d).norm()<=radius) d.push_back(q);} if (d.empty()) { for (const auto& p: raw2) d.emplace_back(p.x,p.y);} 
            std::vector<double> dd; dd.reserve(s.size()); for (const auto& q: s){ Eigen::Vector2d m=iso.linear()*q + iso.translation(); double best=1e9; for (const auto& r: d){ double vv=(r-m).norm(); if (vv<best) best=vv; } dd.push_back(best);} if (dd.empty()) return 1e9; std::sort(dd.begin(),dd.end()); int k=std::max(1,(int)dd.size()/2); double sum=0.0; for (int i=0;i<k;++i) sum+=dd[i]; return sum/k; };
        // whole-cloud robust mse (no center restriction)
        auto points_alignment_mse_all = [&](const Eigen::Isometry2d& iso){
            std::vector<Eigen::Vector2d> s; s.reserve(raw1.size()); for (const auto& p: raw1) s.emplace_back(p.x,p.y);
            std::vector<Eigen::Vector2d> d; d.reserve(raw2.size()); for (const auto& p: raw2) d.emplace_back(p.x,p.y);
            if (s.empty() || d.empty()) return 1e9;
            std::vector<double> dd; dd.reserve(s.size());
            for (const auto& q: s){ Eigen::Vector2d m=iso.linear()*q + iso.translation(); double best=1e9; for (const auto& r: d){ double vv=(r-m).norm(); if (vv<best) best=vv; } dd.push_back(best);} 
            if (dd.empty()) return 1e9; std::sort(dd.begin(),dd.end()); int k=std::max(1,(int)dd.size()/2); double sum=0.0; for (int i=0;i<k;++i) sum+=dd[i]; return sum/k;
        };
        bool has_best=false; int best_inl=0; double best_mse=1e9; double best_pts=1e9; double best_all=1e9; double best_end=1e9; Eigen::Isometry2d best=Eigen::Isometry2d::Identity();
        const bool singleL = (std::min(F1.size(), F2.size()) == 1);
        for (size_t i=0;i<F1.size();++i){
            for (size_t j=0;j<F2.size();++j){
                if (std::abs(F1[i].ldeg - F2[j].ldeg) > 10.0) continue;
                if (singleL) {
                    auto cands = estimate_from_L_candidates(F1[i], F2[j]);
                    for (const auto& iso : cands){
                        double mse; int inl = count_inliers(iso, mse);
                        double mse_pts = points_alignment_mse(iso, F1[i].lc, F2[j].lc, 1.5);
                        double mse_all = points_alignment_mse_all(iso);
                        double e_end = endpoint_score(F1[i], F2[j], iso);
                        if (!has_best || (e_end < best_end) || (std::abs(e_end-best_end)<1e-6 && ((mse_all < best_all) || (std::abs(mse_all-best_all)<1e-6 && (mse_pts < best_pts || (std::abs(mse_pts-best_pts)<1e-6 && (inl > best_inl || (inl==best_inl && mse < best_mse)))))))){
                            has_best=true; best_end=e_end; best_all=mse_all; best_pts=mse_pts; best_inl=inl; best_mse=mse; best=iso;
                        }
                    }
                } else {
                    Eigen::Isometry2d iso = estimate_from_L(F1[i], F2[j]);
                    double mse; int inl = count_inliers(iso, mse);
                    double mse_pts = points_alignment_mse(iso, F1[i].lc, F2[j].lc, 1.5);
                    double mse_all = points_alignment_mse_all(iso);
                    double e_end = endpoint_score(F1[i], F2[j], iso);
                    if (!has_best || inl > best_inl || (inl==best_inl && (e_end < best_end || (std::abs(e_end-best_end)<1e-6 && (mse_pts < best_pts || (std::abs(mse_pts-best_pts)<1e-6 && mse < best_mse)))))){
                        has_best=true; best_inl=inl; best_mse=mse; best_pts=mse_pts; best_all=mse_all; best_end=e_end; best=iso;
                    }
                }
            }
        }
        // fallback: if still not selected (unlikely), use first pair estimate
        if (!has_best) best = estimate_from_L(F1.front(), F2.front());
        return best;
    };

    // Decide dataset type by circle residuals
    // Extract circle centers per map via simple epsilon clustering + Kasa circle fit
    auto extract_circle_centers = [&](const Cloud& cl) -> std::vector<Eigen::Vector2d> {
        const int n = static_cast<int>(cl.size());
        if (n == 0) return {};
        // pairwise distances (O(n^2))
        std::vector<std::vector<int>> neighbors(n);
        const double eps = 0.12; // ~ 12 cm
        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                double dx = cl[i].x - cl[j].x, dy = cl[i].y - cl[j].y;
                if (std::sqrt(dx*dx + dy*dy) <= eps) { neighbors[i].push_back(j); neighbors[j].push_back(i); }
            }
        }
        std::vector<int> label(n, -1);
        int cid = 0;
        for (int i = 0; i < n; ++i) {
            if (label[i] != -1) continue;
            // BFS from i
            std::queue<int> qq; qq.push(i); label[i] = cid; int count = 0;
            while (!qq.empty()) { int u = qq.front(); qq.pop(); ++count; for (int v : neighbors[u]) if (label[v] == -1) { label[v] = cid; qq.push(v); } }
            cid++;
        }
        // For each cluster, if size small enough (< 2*PI*R/spacing), fit circle
        std::vector<Eigen::Vector2d> centers;
        for (int c = 0; c < cid; ++c) {
            std::vector<int> idx;
            for (int i = 0; i < n; ++i) if (label[i] == c) idx.push_back(i);
            if ((int)idx.size() < 12) continue; // need enough points
            // fit Kasa on this cluster
            Eigen::MatrixXd A(idx.size(), 3);
            Eigen::VectorXd b(idx.size());
            for (size_t k = 0; k < idx.size(); ++k) {
                double x = cl[idx[k]].x, y = cl[idx[k]].y;
                A(k,0) = x; A(k,1) = y; A(k,2) = 1.0;
                b(k) = -(x*x + y*y);
            }
            Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(b);
            double A_ = coeffs(0), B_ = coeffs(1), C_ = coeffs(2);
            double cx = -A_/2.0, cy = -B_/2.0; double R2 = (A_*A_ + B_*B_)/4.0 - C_;
            if (R2 <= 0) continue; double R = std::sqrt(R2);
            // residual normalized by R
            double sum = 0.0; for (int id : idx) { double dx = cl[id].x - cx, dy = cl[id].y - cy; double r = std::sqrt(dx*dx + dy*dy); sum += std::abs(r - R); }
            double resn = (sum / idx.size()) / std::max(1e-9, R);
            if (R >= 0.015 && R <= 0.05 && resn < 0.08) centers.emplace_back(cx, cy);
        }
        return centers;
    };

    auto estimate_from_centers = [](const std::vector<Eigen::Vector2d>& S, const std::vector<Eigen::Vector2d>& D) -> Eigen::Isometry2d {
        // RANSAC on two-point hypotheses
        if (S.size() < 2 || D.size() < 2) {
            Eigen::Isometry2d id = Eigen::Isometry2d::Identity(); return id;
        }
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> ds(0, (int)S.size()-1), dd(0, (int)D.size()-1);
        auto compute_iso = [](const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& q1, const Eigen::Vector2d& q2) {
            Eigen::Vector2d vs = p2 - p1; Eigen::Vector2d vd = q2 - q1;
            double ang = std::atan2(vd.y(), vd.x()) - std::atan2(vs.y(), vs.x());
            Eigen::Rotation2D<double> R(ang);
            Eigen::Isometry2d iso = Eigen::Isometry2d::Identity(); iso.linear() = R.toRotationMatrix(); iso.translation() = q1 - iso.linear() * p1; return iso;
        };
        auto count_inliers = [&](const Eigen::Isometry2d& iso, std::vector<std::pair<int,int>>& pairs) {
            const double tol = 0.15; pairs.clear(); std::vector<char> used(D.size(), 0);
            for (size_t i = 0; i < S.size(); ++i) {
                Eigen::Vector2d m = iso.linear()*S[i] + iso.translation();
                int bestj = -1; double bestd = 1e9;
                for (size_t j = 0; j < D.size(); ++j) {
                    if (used[j]) continue; double d = (D[j]-m).norm(); if (d < bestd) { bestd = d; bestj = (int)j; }
                }
                if (bestj >= 0 && bestd <= tol) { used[bestj] = 1; pairs.emplace_back((int)i, bestj); }
            }
            return (int)pairs.size();
        };
        int best_inl = -1; Eigen::Isometry2d best_iso = Eigen::Isometry2d::Identity(); std::vector<std::pair<int,int>> best_pairs;
        for (int it = 0; it < 300; ++it) {
            int i1 = ds(rng), i2 = ds(rng); if (i1 == i2) continue; int j1 = dd(rng), j2 = dd(rng); if (j1 == j2) continue;
            Eigen::Isometry2d iso = compute_iso(S[i1], S[i2], D[j1], D[j2]);
            std::vector<std::pair<int,int>> pairs; int ninl = count_inliers(iso, pairs);
            if (ninl > best_inl) { best_inl = ninl; best_iso = iso; best_pairs = std::move(pairs); }
        }
        // refine via Procrustes on inliers
        if (best_inl >= 2) {
            Eigen::Vector2d cs(0,0), cd(0,0);
            for (auto& pr : best_pairs) { cs += S[pr.first]; cd += D[pr.second]; }
            cs /= (double)best_pairs.size(); cd /= (double)best_pairs.size();
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            for (auto& pr : best_pairs) {
                Eigen::Vector2d xs = S[pr.first] - cs; Eigen::Vector2d xd = D[pr.second] - cd; H += xs * xd.transpose();
            }
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
            if (R.determinant() < 0) { Eigen::Matrix2d V = svd.matrixV(); V.col(1) *= -1; R = V * svd.matrixU().transpose(); }
            Eigen::Isometry2d iso = Eigen::Isometry2d::Identity(); iso.linear() = R; iso.translation() = cd - R * cs; return iso;
        }
        return best_iso;
    };

    // Precompute features for all maps
    std::vector<std::vector<Eigen::Vector2d>> circle_centers(N);
    std::vector<std::vector<LFeat>> Lfeats(N);
    for (size_t i = 0; i < N; ++i) {
        circle_centers[i] = extract_circle_centers(m_map_info[i].map);
        Lfeats[i] = extract_L_features(m_map_info[i].map);
        std::cerr << "[L-debug] map" << (i+1) << " L-features: " << Lfeats[i].size() << " circles: " << circle_centers[i].size() << "\n";
    }

    // For BFS disambiguation, keep candidate transforms per edge (v->u)
    std::unordered_map<long long, std::vector<Eigen::Isometry2d>> edge_vu_cands; // key = (u<<32)|v
    std::unordered_map<long long, Eigen::Isometry2d> edge_uv; // representative (best) for fallback
    auto key_uv = [](int u, int v) -> long long { return (static_cast<long long>(u) << 32) ^ static_cast<unsigned long long>(v); };

    auto choose_best_L_pair = [&](const std::vector<LFeat>& F1, const std::vector<LFeat>& F2, const Cloud& raw1, const Cloud& raw2) -> std::pair<int,int> {
        int bi=-1, bj=-1; bool has=false; int best_inl=0; double best_mse=1e9; double best_pts=1e9;
        auto count_inliers = [&](const Eigen::Isometry2d& iso){
            const double tol=0.3; int used_ct=0; std::vector<char> used(F2.size(),0);
            for (size_t i=0;i<F1.size();++i){ Eigen::Vector2d m = iso.linear()*F1[i].lc + iso.translation(); int best=-1; double bd=1e9; for (size_t j=0;j<F2.size();++j){ if (used[j]) continue; double d=(F2[j].lc - m).norm(); if (d<bd){ bd=d; best=(int)j; } } if (best>=0 && bd<=tol){ used[best]=1; used_ct++; } }
            return used_ct;
        };
        auto points_alignment_mse_local = [&](const Eigen::Isometry2d& iso, const Eigen::Vector2d& cs, const Eigen::Vector2d& cd){
            std::vector<Eigen::Vector2d> s; for (const auto& p: raw1){ Eigen::Vector2d q(p.x,p.y); if ((q-cs).norm()<=1.5) s.push_back(q);} if (s.empty()) { for (const auto& p: raw1) s.emplace_back(p.x,p.y);} 
            std::vector<Eigen::Vector2d> d; for (const auto& p: raw2){ Eigen::Vector2d q(p.x,p.y); if ((q-cd).norm()<=1.5) d.push_back(q);} if (d.empty()) { for (const auto& p: raw2) d.emplace_back(p.x,p.y);} 
            std::vector<double> dd; dd.reserve(s.size()); for (const auto& q: s){ Eigen::Vector2d m=iso.linear()*q + iso.translation(); double best=1e9; for (const auto& r: d){ double vv=(r-m).norm(); if (vv<best) best=vv; } dd.push_back(best);} if (dd.empty()) return 1e9; std::sort(dd.begin(),dd.end()); int k=std::max(1,(int)dd.size()/2); double sum=0.0; for (int i=0;i<k;++i) sum+=dd[i]; return sum/k; };
        for (size_t i=0;i<F1.size();++i){
            for (size_t j=0;j<F2.size();++j){
                if (std::abs(F1[i].ldeg - F2[j].ldeg) > 10.0) continue;
                Eigen::Isometry2d iso = estimate_from_L(F1[i], F2[j]);
                int inl = count_inliers(iso);
                double pts = points_alignment_mse_local(iso, F1[i].lc, F2[j].lc);
                if (!has || inl > best_inl || (inl==best_inl && pts < best_pts)){
                    has=true; best_inl=inl; best_pts=pts; bi=(int)i; bj=(int)j;
                }
            }
        }
        if (!has) return {0,0};
        return {bi,bj};
    };
    for (size_t u = 0; u < N; ++u) {
        for (int v : m_adj[u]) {
            if (edge_uv.find(key_uv(u, v)) != edge_uv.end()) continue;
            Eigen::Isometry2d uv;
            if (circle_centers[u].size() >= 2 && circle_centers[v].size() >= 2) {
                uv = estimate_from_centers(circle_centers[u], circle_centers[v]);
                edge_vu_cands[key_uv(v,u)] = { invertIso2D(uv) };
            } else if (!Lfeats[u].empty() && !Lfeats[v].empty()) {
                // if single-L on the edge, keep both orientation candidates for BFS disambiguation
                if (std::min(Lfeats[u].size(), Lfeats[v].size()) == 1) {
                    auto [iu,jv] = choose_best_L_pair(Lfeats[u], Lfeats[v], m_map_info[u].map, m_map_info[v].map);
                    auto cands_uv = estimate_from_L_candidates(Lfeats[u][iu], Lfeats[v][jv]);
                    // choose a representative by local scoring
                    double best_pts = 1e9; int best_k = 0;
                    for (int k=0;k<(int)cands_uv.size();++k){
                        auto iso = cands_uv[k];
                        // local corner region robust mse as score
                        std::vector<Eigen::Vector2d> s; for (const auto& p: m_map_info[u].map){ Eigen::Vector2d q(p.x,p.y); if ((q-Lfeats[u][iu].lc).norm()<=1.5) s.push_back(q);} if (s.empty()) for (const auto& p: m_map_info[u].map) s.emplace_back(p.x,p.y);
                        std::vector<Eigen::Vector2d> d; for (const auto& p: m_map_info[v].map){ Eigen::Vector2d q(p.x,p.y); if ((q-Lfeats[v][jv].lc).norm()<=1.5) d.push_back(q);} if (d.empty()) for (const auto& p: m_map_info[v].map) d.emplace_back(p.x,p.y);
                        std::vector<double> dd; for (const auto& q: s){ Eigen::Vector2d m=iso.linear()*q + iso.translation(); double best=1e9; for (const auto& r: d){ double vv=(r-m).norm(); if (vv<best) best=vv; } dd.push_back(best);} if (dd.empty()) continue; std::sort(dd.begin(),dd.end()); int kk=std::max(1,(int)dd.size()/2); double sum=0.0; for (int ii=0; ii<kk; ++ii) sum+=dd[ii]; double score = sum/kk;
                        if (score < best_pts){ best_pts = score; best_k = k; }
                    }
                    uv = cands_uv[best_k];
                    auto cands_vu = std::vector<Eigen::Isometry2d>{ invertIso2D(cands_uv[0]), invertIso2D(cands_uv[1]) };
                    edge_vu_cands[key_uv(v,u)] = cands_vu;
                } else {
                    uv = estimate_from_Ls(Lfeats[u], Lfeats[v], m_map_info[u].map, m_map_info[v].map);
                    edge_vu_cands[key_uv(v,u)] = { invertIso2D(uv) };
                }
            } else {
                // fallback: ICP for robustness
                Cloud::ConstPtr src(new Cloud(m_map_info[u].map));
                Cloud::ConstPtr dst(new Cloud(m_map_info[v].map));
                auto [ok, T] = icp2D(src, dst, 2.0, 100, 1e-8, 1e-8);
                (void)ok; uv = toIso2D(T);
                edge_vu_cands[key_uv(v,u)] = { invertIso2D(uv) };
            }
            edge_uv[key_uv(u, v)] = uv;
            edge_uv[key_uv(v, u)] = invertIso2D(uv);
        }
    }

    // BFS from root=0 to compute transforms to reference
    m_transform.assign(N, Eigen::Isometry2d::Identity());
    std::vector<char> visited(N, 0);
    std::queue<int> q;
    visited[0] = 1;
    m_transform[0] = Eigen::Isometry2d::Identity();
    q.push(0);
    auto robust_mse_between = [&](const Eigen::Isometry2d& Ta, const Cloud& A, const Eigen::Isometry2d& Tb, const Cloud& B){
        if (A.empty() || B.empty()) return 1e9;
        std::vector<Eigen::Vector2d> a; a.reserve(A.size()); for (const auto& p: A) a.emplace_back(p.x,p.y);
        std::vector<Eigen::Vector2d> b; b.reserve(B.size()); for (const auto& p: B) b.emplace_back(p.x,p.y);
        std::vector<double> dd; dd.reserve(a.size());
        for (const auto& q2 : a){ Eigen::Vector2d u2 = Ta.linear()*q2 + Ta.translation(); double best=1e9; for (const auto& r2 : b){ Eigen::Vector2d v2 = Tb.linear()*r2 + Tb.translation(); double d=(v2-u2).norm(); if (d<best) best=d; } dd.push_back(best);} if (dd.empty()) return 1e9; std::sort(dd.begin(),dd.end()); int k=std::max(1,(int)dd.size()/2); double s=0.0; for (int i=0;i<k;++i) s+=dd[i]; return s/k; };

    auto robust_mse_to_target = [&](const Eigen::Isometry2d& Ta, const Cloud& A, const std::vector<Eigen::Vector2d>& targetPts){
        if (A.empty() || targetPts.empty()) return 1e9;
        std::vector<double> dd; dd.reserve(A.size());
        for (const auto& p : A){ Eigen::Vector2d u2 = Ta.linear()*Eigen::Vector2d(p.x,p.y) + Ta.translation(); double best=1e9; for (const auto& t : targetPts){ double d=(t-u2).norm(); if (d<best) best=d; } dd.push_back(best);} if (dd.empty()) return 1e9; std::sort(dd.begin(),dd.end()); int k=std::max(1,(int)dd.size()/2); double s=0.0; for (int i=0;i<k;++i) s+=dd[i]; return s/k; };

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : m_adj[u]) {
            if (visited[v]) continue;
            auto it = edge_vu_cands.find(key_uv(v, u)); // v->u candidate transforms
            if (it == edge_vu_cands.end()) {
                std::cerr << "Missing edge transform for (" << v << "," << u << ")" << std::endl;
                continue;
            }
            // Build aggregate target from all visited nodes (global consistency)
            std::vector<Eigen::Vector2d> targetPts; targetPts.reserve(1024);
            for (size_t k=0;k<N;++k){ if (!visited[k]) continue; for (const auto& p : m_map_info[k].map){ Eigen::Vector2d q(p.x,p.y); Eigen::Vector2d tpt = m_transform[k].linear()*q + m_transform[k].translation(); targetPts.push_back(tpt); } }
            // choose best candidate by global consistency
            double best_err = 1e9; Eigen::Isometry2d best_v1 = Eigen::Isometry2d::Identity(); bool chosen=false;
            for (const auto& vu : it->second){
                Eigen::Isometry2d v1 = composeIso2D(vu, m_transform[u]);
                double e = robust_mse_to_target(v1, m_map_info[v].map, targetPts);
                if (!chosen || e < best_err){ chosen=true; best_err=e; best_v1=v1; }
            }
            if (!chosen) best_v1 = composeIso2D(edge_uv[key_uv(v,u)], m_transform[u]);
            m_transform[v] = best_v1;
            visited[v] = 1;
            q.push(v);
        }
    }

    // Post-process: resolve residual 180° ambiguities by global consistency
    // For each node (except root), consider flipping around its mapped centroid and keep if global robust MSE improves.
    std::vector<Eigen::Vector2d> allPts; allPts.reserve(2048);
    for (size_t k=0;k<N;++k){ for (const auto& p : m_map_info[k].map){ Eigen::Vector2d q(p.x,p.y); Eigen::Vector2d tpt = m_transform[k].linear()*q + m_transform[k].translation(); allPts.push_back(tpt); } }
    for (size_t i=1;i<N;++i){
        // build target excluding i
        std::vector<Eigen::Vector2d> target; target.reserve(allPts.size());
        for (size_t k=0;k<N;++k){ if (k==i) continue; for (const auto& p : m_map_info[k].map){ Eigen::Vector2d q(p.x,p.y); Eigen::Vector2d tpt = m_transform[k].linear()*q + m_transform[k].translation(); target.push_back(tpt); } }
        if (target.empty()) continue;
        // centroid in source and mapped
        Eigen::Vector2d c_src(0,0); if (!m_map_info[i].map.empty()){ for (const auto& p:m_map_info[i].map){ c_src.x()+=p.x; c_src.y()+=p.y; } c_src/= (double)m_map_info[i].map.size(); }
        Eigen::Vector2d c_map = m_transform[i].linear()*c_src + m_transform[i].translation();
        Eigen::Rotation2D<double> R180(M_PI);
        Eigen::Isometry2d Tflip = Eigen::Isometry2d::Identity();
        Tflip.linear() = R180.toRotationMatrix() * m_transform[i].linear();
        Tflip.translation() = -m_transform[i].translation() + 2.0 * c_map;
        double e_keep = robust_mse_to_target(m_transform[i], m_map_info[i].map, target);
        double e_flip = robust_mse_to_target(Tflip, m_map_info[i].map, target);
        if (e_flip + 1e-9 < e_keep) m_transform[i] = Tflip;
    }
    return true;
}

} // namespace mc
