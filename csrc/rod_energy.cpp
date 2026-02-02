#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <numeric>
#include <limits>
#include <cassert>
#include <iostream>
#include <limits>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

// Exported API (students may extend, but autograder only requires Python-level RodEnergy.value_and_grad).
// x: length 3N (xyzxyz...)
// grad_out: length 3N
// Periodic indexing enforces a closed loop.
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double kc,     // confinement strength
    double eps,    // WCA epsilon
    double sigma,  // WCA sigma
    double* energy_out,
    double* grad_out
) {
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };
    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    auto computeClosest = [&](int i, int j) -> std::array<double,2> {
        std::array<double,3> p = {get(i,0), get(i,1), get(i,2)};
        std::array<double,3> q = {get(j,0), get(j,1), get(j,2)};
        std::array<double,3> d1 = {get(i+1,0)-get(i,0), get(i+1,1)-get(i,1), get(i+1,2)-get(i,2)};
        std::array<double,3> d2 = {get(j+1,0)-get(j,0), get(j+1,1)-get(j,1), get(j+1,2)-get(j,2)};

        double a = std::inner_product(d1.begin(),d1.end(),d1.begin(),0.0);
        double b = std::inner_product(d1.begin(),d1.end(),d2.begin(),0.0);
        double c = std::inner_product(d2.begin(),d2.end(),d2.begin(),0.0);
        std::array<double,3> r;
        for(int k=0;k<3;k++) r[k] = p[k]-q[k];
        double d = std::inner_product(d1.begin(),d1.end(),r.begin(),0.0);
        double e = std::inner_product(d2.begin(),d2.end(),r.begin(),0.0);

        double det = a*c - b*b;
        double u,v;
        if(det > 1e-12) {
            u = (b*e - c*d)/det;
            v = (a*e - b*d)/det;
        } else {
            u = 0; v = 0; // parallel, fallback
        }
        u = std::clamp(u,0.0,1.0);
        v = std::clamp(v,0.0,1.0);
        return {u,v};
    };

    auto pointPointWCA = [&](int a, int b)
    {
        double rx = get(a,0) - get(b,0);
        double ry = get(a,1) - get(b,1);
        double rz = get(a,2) - get(b,2);
    
        double dist = std::sqrt(rx*rx + ry*ry + rz*rz);
        dist = std::max(dist, 1e-12);
    
        const double cutoff = std::pow(2.0, 1.0/6.0) * sigma;
        if (dist >= cutoff) return;
    
        double invd = 1.0 / dist;
        double s2 = (sigma * invd) * (sigma * invd);
        double s6 = s2 * s2 * s2;
    
        double Uwca = 4 * eps * (s6*s6 - s6) + eps;
        E += Uwca;
    
        double forceMag = 24 * eps * invd * (2*s6*s6 - s6);
        double fx = forceMag * rx * invd;
        double fy = forceMag * ry * invd;
        double fz = forceMag * rz * invd;
    
        addg(a,0,  fx); addg(a,1,  fy); addg(a,2,  fz);
        addg(b,0, -fx); addg(b,1, -fy); addg(b,2, -fz);
    };

    auto pointSegmentWCA = [&](int p, int s0, int s1)
    {
        double sx = get(s1,0) - get(s0,0);
        double sy = get(s1,1) - get(s0,1);
        double sz = get(s1,2) - get(s0,2);

        double px = get(p,0) - get(s0,0);
        double py = get(p,1) - get(s0,1);
        double pz = get(p,2) - get(s0,2);

        double seg2 = sx*sx + sy*sy + sz*sz;
        if (seg2 < 1e-14) return;

        double t = (px*sx + py*sy + pz*sz) / seg2;
        t = std::clamp(t, 0.0, 1.0);

        double cx = get(s0,0) + t*sx;
        double cy = get(s0,1) + t*sy;
        double cz = get(s0,2) + t*sz;

        double rx = get(p,0) - cx;
        double ry = get(p,1) - cy;
        double rz = get(p,2) - cz;

        double dist = std::sqrt(rx*rx + ry*ry + rz*rz);
        dist = std::max(dist, 1e-12);

        const double cutoff = std::pow(2.0, 1.0/6.0) * sigma;
        if (dist >= cutoff) return;

        double invd = 1.0 / dist;
        double s2 = (sigma * invd) * (sigma * invd);
        double s6 = s2 * s2 * s2;

        double Uwca = 4 * eps * (s6*s6 - s6) + eps;
        E += Uwca;

        double forceMag = 24 * eps * invd * (2*s6*s6 - s6);
        double fx = forceMag * rx * invd;
        double fy = forceMag * ry * invd;
        double fz = forceMag * rz * invd;

        // point gets full force
        addg(p,0,  fx); addg(p,1,  fy); addg(p,2,  fz);

        // segment endpoints share opposite force
        addg(s0,0, -fx*(1-t));
        addg(s0,1, -fy*(1-t));
        addg(s0,2, -fz*(1-t));

        addg(s1,0, -fx*t);
        addg(s1,1, -fy*t);
        addg(s1,2, -fz*t);
    };




    // ---- Bending: kb * ||x_{i+1} - 2 x_i + x_{i-1}||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double b = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * b * b;
            const double c = 2.0 * kb * b;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // ---- Stretching: ks * (||x_{i+1}-x_i|| - l0)^2
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12);
        double diff = r - l0;
        E += ks * diff * diff;

        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // ---- Confinement: kc * sum ||x_i||^2
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            double xi = get(i,d);
            E += kc * xi * xi;
            addg(i,d, 2.0 * kc * xi);
        }
    }

    // ---- Segment–segment WCA self-avoidance ----
    //
    // For each non-adjacent segment pair (i,i+1) and (j,j+1):
    //  1) Compute closest points parameters u*, v* in [0,1]
    //  2) Compute r = p_i(u*) - p_j(v*),  d = ||r||
    //  3) If d < 2^(1/6)*sigma:
    //       U(d) = 4 eps [ (sigma/d)^12 - (sigma/d)^6 ] + eps
    //       Accumulate E += U(d)
    //       Accumulate gradient to endpoints x_i, x_{i+1}, x_j, x_{j+1}
    //
    // Exclusions: skip adjacent segments (including wrap neighbors).
    //
    // IMPORTANT: include dependence of (u*, v*) on endpoints in gradient.

    for(int i=0;i<N;i++)
    {
        for(int j=i+1;j<N;j++)
        {
            int i0 = i;
            int i1 = idx(i+1);
            int j0 = j;
            int j1 = idx(j+1);
            if( i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1 )
                continue; // skip adjacent segments

            auto optimal = computeClosest(i,j);
            double u = optimal[0];
            double v = optimal[1];
            bool u_clamped = (u == 0.0 || u == 1.0);
            bool v_clamped = (v == 0.0 || v == 1.0);

            if (u_clamped && v_clamped)
            {
                int pi = (u == 0.0) ? i : i+1;
                int pj = (v == 0.0) ? j : j+1;
                pointPointWCA(pi, pj);
            }
            else if (u_clamped)
            {
                int pi = (u == 0.0) ? i : i+1;
                pointSegmentWCA(pi, j, j+1);
            }
            else if (v_clamped)
            {
                int pj = (v == 0.0) ? j : j+1;
                pointSegmentWCA(pj, i, i+1);
            }
            else
            {
                double dist;
                double tempx = 0.0;
                double tempy = 0.0;
                double tempz = 0.0;

                tempx = (get(i,0) + u*(get(i+1,0) - get(i,0))) - (get(j,0) + v*(get(j+1,0) - get(j,0)));
                tempy = (get(i,1) + u*(get(i+1,1) - get(i,1))) - (get(j,1) + v*(get(j+1,1) - get(j,1))); 
                tempz = (get(i,2) + u*(get(i+1,2) - get(i,2))) - (get(j,2) + v*(get(j+1,2) - get(j,2)));

                dist = std::sqrt((tempx)*(tempx) + (tempy)*(tempy) + (tempz)*(tempz));
                dist = std::max(dist, 1e-12);
                const double cutoff = std::pow(2.0, 1.0/6.0) * sigma;
                if (dist < cutoff)
                {
                    double invd = 1.0 / dist;
                    double s2 = (sigma * invd) * (sigma * invd);
                    double s6 = s2 * s2 * s2;
                    E += 4 * eps * (s6*s6 - s6) + eps;

                    // if (4 * eps * (s6*s6 - s6) + eps > 1e5)
                    // {
                    //     std::cout<<"Large WCA energy detected!" << std::endl;
                    //     std::cout<< "E WCA contribution: " << 4 * eps * (s6*s6 - s6) + eps << " at dist " << dist << std::endl;
                    //     std::cout<< "  between segments (" << i << "," << i+1 << ") and (" << j << "," << j+1 << ")" << std::endl;
                    //     std::cout<< "  with parameters u=" << u << ", v=" << v << std::endl;
                    //     std::cout<< "-------------------------------------------" << std::endl;
                    // }
                    double forceMag = 24 * eps * invd * (2*s6*s6 - s6);
                    double fx = forceMag * (tempx * invd);
                    double fy = forceMag * (tempy * invd);
                    double fz = forceMag * (tempz * invd);

                    addg(i,   0,  -fx * (1-u));
                    addg(i,   1,  -fy * (1-u));
                    addg(i,   2,  -fz * (1-u));

                    addg(i+1, 0,  -fx * u);
                    addg(i+1, 1,  -fy * u);
                    addg(i+1, 2,  -fz * u);

                    addg(j,   0, fx * (1-v));
                    addg(j,   1, fy * (1-v));
                    addg(j,   2, fz * (1-v));

                    addg(j+1, 0, fx * v);
                    addg(j+1, 1, fy * v);
                    addg(j+1, 2, fz * v);



                }

            }

        }
    }
    

    *energy_out = E;
}

} // extern "C"