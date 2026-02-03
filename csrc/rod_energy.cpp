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
        if(det > 1e-6) {
            u = (b*e - c*d)/det;
            v = (a*e - b*d)/det;
        }
        else {
            // Parallel case: find the closest point on segment j to the start of segment i
            // Since det ~ 0, we can't use the system of equations.
            // We project vector r (p-q) onto d2 to find the position v.
            u = 0.0;
            double d2_mag_sq = c; // c is d2.d2
            if (d2_mag_sq > 1e-9) {
                // v = dot(p - q, d2) / dot(d2, d2)
                // Note: r = p - q, so v = dot(r, d2) / c
                v = e / c; 
            } else {
                v = 0.0; // Segment j is a single point
            }
        }
        u = std::clamp(u,0.0,1.0);
        v = std::clamp(v,0.0,1.0);
        return {u,v};
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
    const double cutoff = std::pow(2.0, 1.0/6.0) * sigma;
    const double cutoffSq = cutoff * cutoff;

    for (int i = 0; i < N; ++i) {
        // j = i + 2 ensures we skip the current segment and the immediate neighbor
        for (int j = i + 3; j < N; ++j) {

            // Special case: if i is the first segment and j is the last, 
            // they share node 0. Skip them.
            if (i == 0 && j == N - 1) continue;

            auto optimal = computeClosest(i, j);
            double u = optimal[0];
            double v = optimal[1];

            // IMPORTANT: Use idx() for the +1 nodes to prevent out-of-bounds
            int node_i0 = i;
            int node_i1 = idx(i + 1);
            int node_j0 = j;
            int node_j1 = idx(j + 1);

            // Position on segment i
            double pix = get(node_i0, 0) + u * (get(node_i1, 0) - get(node_i0, 0));
            double piy = get(node_i0, 1) + u * (get(node_i1, 1) - get(node_i0, 1));
            double piz = get(node_i0, 2) + u * (get(node_i1, 2) - get(node_i0, 2));

            // Position on segment j
            double pjx = get(node_j0, 0) + v * (get(node_j1, 0) - get(node_j0, 0));
            double pjy = get(node_j0, 1) + v * (get(node_j1, 1) - get(node_j0, 1));
            double pjz = get(node_j0, 2) + v * (get(node_j1, 2) - get(node_j0, 2));

            double dx = pix - pjx;
            double dy = piy - pjy;
            double dz = piz - pjz;

            double distSq = dx*dx + dy*dy + dz*dz;

            if (distSq < cutoffSq) {
                double dist = std::sqrt(distSq);
                dist = std::max(dist, 1e-9); 

                double invd = 1.0 / dist;
                double s2 = (sigma * invd) * (sigma * invd);
                double s6 = s2 * s2 * s2;

                E += 4.0 * eps * (s6 * s6 - s6) + eps;

                double forceMag = 24.0 * eps * invd * (2.0 * s6 * s6 - s6);
                double fx = forceMag * (dx * invd);
                double fy = forceMag * (dy * invd);
                double fz = forceMag * (dz * invd);

                // Use the node variables to ensure idx() is applied
                addg(node_i0, 0,  fx * (1.0 - u));
                addg(node_i0, 1,  fy * (1.0 - u));
                addg(node_i0, 2,  fz * (1.0 - u));

                addg(node_i1, 0,  fx * u);
                addg(node_i1, 1,  fy * u);
                addg(node_i1, 2,  fz * u);

                addg(node_j0, 0, -fx * (1.0 - v));
                addg(node_j0, 1, -fy * (1.0 - v));
                addg(node_j0, 2, -fz * (1.0 - v));

                addg(node_j1, 0, -fx * v);
                addg(node_j1, 1, -fy * v);
                addg(node_j1, 2, -fz * v);
            }
        }
    }
    

    *energy_out = E;
}

} // extern "C"