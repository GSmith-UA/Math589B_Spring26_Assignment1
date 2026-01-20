#include <cmath>
#include <algorithm>

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

    const double rcut = std::pow(2.0, 1.0/6.0) * sigma;
    const double rcut2 = rcut * rcut;
    const double small = 1e-12;

    // helper lambda to read point into array
    auto read_point = [&](int i, double p[3]) {
        p[0] = get(i,0); p[1] = get(i,1); p[2] = get(i,2);
    };

    for (int i = 0; i < N; ++i) {
        for (int j = i+2; j < N; ++j) {
            // skip wrap adjacent (i==0 and j==N-1)
            if (i==0 && j==N-1) continue;

            // endpoints a = x_i, b = x_{i+1}, c = x_j, d = x_{j+1}
            double a[3], bpt[3], cpt[3], dpt[3];
            read_point(i, a);
            read_point(i+1, bpt);
            read_point(j, cpt);
            read_point(j+1, dpt);

            double s[3], t[3], r0[3];
            for (int d = 0; d < 3; ++d) {
                s[d] = bpt[d] - a[d];
                t[d] = dpt[d] - cpt[d];
                r0[d] = a[d] - cpt[d];
            }

            double A = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
            double B = s[0]*t[0] + s[1]*t[1] + s[2]*t[2];
            double C = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
            double D = s[0]*r0[0] + s[1]*r0[1] + s[2]*r0[2];
            double E_ = t[0]*r0[0] + t[1]*r0[1] + t[2]*r0[2];

            double u = 0.0, v = 0.0;
            bool use_full_deriv = true;

            double denom = A*C - B*B;
            if (denom > 1e-14) {
                u = (B*E_ - C*D) / denom;
                v = (A*E_ - B*D) / denom;
            } else {
                // nearly parallel: project endpoints conservatively
                u = 0.0;
                if (C > small) v = (t[0]*r0[0] + t[1]*r0[1] + t[2]*r0[2]) / C;
                else v = 0.0;
            }

            // clamp and if clamped recompute the other parameter optimally
            bool u_clamped = false, v_clamped = false;
            double u_uncl = u, v_uncl = v;
            if (u < 0.0) { u = 0.0; u_clamped = true; }
            else if (u > 1.0) { u = 1.0; u_clamped = true; }
            if (v < 0.0) { v = 0.0; v_clamped = true; }
            else if (v > 1.0) { v = 1.0; v_clamped = true; }

            if (u_clamped && !v_clamped) {
                // recompute optimal v given u
                double numer = t[0]*(r0[0] + u*s[0]) + t[1]*(r0[1] + u*s[1]) + t[2]*(r0[2] + u*s[2]);
                if (C > small) v = numer / C;
                else v = 0.0;
                v = std::min(1.0, std::max(0.0, v));
            } else if (v_clamped && !u_clamped) {
                double numer = s[0]*( -r0[0] + v*t[0]) + s[1]*( -r0[1] + v*t[1]) + s[2]*( -r0[2] + v*t[2]);
                if (A > small) u = numer / A;
                else u = 0.0;
                u = std::min(1.0, std::max(0.0, u));
            } else if (u_clamped && v_clamped) {
                // both clamped, fine
                use_full_deriv = false; // derivatives of u,v wrt endpoints are zero (they are on boundaries)
            } else {
                // neither clamped: keep u,v from unconstrained; proceed with full deriv
            }

            // recompute closest points and r
            double p[3], q[3], rvec[3];
            for (int d = 0; d < 3; ++d) {
                p[d] = a[d] + u * s[d];
                q[d] = cpt[d] + v * t[d];
                rvec[d] = p[d] - q[d];
            }
            double d2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
            if (d2 >= rcut2) continue;
            double dlen = std::sqrt(std::max(d2, 1e-16));

            // WCA potential and derivative
            double invd = 1.0 / dlen;
            double sr = sigma * invd;
            double sr2 = sr*sr;
            double sr6 = sr2*sr2*sr2;
            double sr12 = sr6*sr6;
            double U = 4.0 * eps * (sr12 - sr6) + eps;
            // dU/dd:
            double dU = 4.0 * eps * ( -12.0 * sr12 * invd + 6.0 * sr6 * invd );
            // (since derivative of (sigma/d)^n is -n * sigma^n / d^{n+1}, combined with prefactors)
            E += U;

            // compute helper scalars
            double dot_sr = s[0]*rvec[0] + s[1]*rvec[1] + s[2]*rvec[2];
            double dot_tr = t[0]*rvec[0] + t[1]*rvec[1] + t[2]*rvec[2];

            // compute derivatives du/dA,B,C,D,E and dv/d...
            double du_dA=0, du_dB=0, du_dC=0, du_dD=0, du_dE=0;
            double dv_dA=0, dv_dB=0, dv_dC=0, dv_dD=0, dv_dE=0;
            if (use_full_deriv && denom > 1e-14) {
                double numu = -C*D + B*E_;
                double numv = A*E_ - B*D;
                double ddenom_dA = C;
                double ddenom_dB = -2.0*B;
                double ddenom_dC = A;
                double ddenom_dD = 0.0;
                double ddenom_dE = 0.0;

                // numu derivatives
                double dnumu_dA = 0.0;
                double dnumu_dB = E_;
                double dnumu_dC = -D;
                double dnumu_dD = -C;
                double dnumu_dE = B;

                double dnumv_dA = E_;
                double dnumv_dB = -D;
                double dnumv_dC = 0.0;
                double dnumv_dD = -B;
                double dnumv_dE = A;

                double denom2 = denom*denom;

                du_dA = (dnumu_dA * denom - numu * ddenom_dA) / denom2;
                du_dB = (dnumu_dB * denom - numu * ddenom_dB) / denom2;
                du_dC = (dnumu_dC * denom - numu * ddenom_dC) / denom2;
                du_dD = (dnumu_dD * denom - numu * ddenom_dD) / denom2;
                du_dE = (dnumu_dE * denom - numu * ddenom_dE) / denom2;

                dv_dA = (dnumv_dA * denom - numv * ddenom_dA) / denom2;
                dv_dB = (dnumv_dB * denom - numv * ddenom_dB) / denom2;
                dv_dC = (dnumv_dC * denom - numv * ddenom_dC) / denom2;
                dv_dD = (dnumv_dD * denom - numv * ddenom_dD) / denom2;
                dv_dE = (dnumv_dE * denom - numv * ddenom_dE) / denom2;
            } else {
                // when degenerate or clamped we will set parameter derivatives to zero
                du_dA = du_dB = du_dC = du_dD = du_dE = 0.0;
                dv_dA = dv_dB = dv_dC = dv_dD = dv_dE = 0.0;
            }

            // Now compute du/d(endpoint) and dv/d(endpoint) vectors using chain rule
            // Precompute vector partials of A,B,C,D,E wrt endpoints a,b,c,d
            // For endpoint a:
            double dA_da[3] = { -2.0*s[0], -2.0*s[1], -2.0*s[2] };
            double dB_da[3] = { -t[0], -t[1], -t[2] };
            double dC_da[3] = { 0.0, 0.0, 0.0 };
            double dD_da[3] = { -r0[0] + s[0], -r0[1] + s[1], -r0[2] + s[2] };
            double dE_da[3] = { t[0], t[1], t[2] };

            double dA_db[3] = {  2.0*s[0],  2.0*s[1],  2.0*s[2] };
            double dB_db[3] = {  t[0],  t[1],  t[2] };
            double dC_db[3] = { 0.0, 0.0, 0.0 };
            double dD_db[3] = { r0[0], r0[1], r0[2] };
            double dE_db[3] = { 0.0, 0.0, 0.0 };

            double dA_dc[3] = { 0.0, 0.0, 0.0 };
            double dB_dc[3] = { -s[0], -s[1], -s[2] };
            double dC_dc[3] = { -2.0*t[0], -2.0*t[1], -2.0*t[2] };
            double dD_dc[3] = { -s[0], -s[1], -s[2] };
            double dE_dc[3] = { -r0[0] - t[0], -r0[1] - t[1], -r0[2] - t[2] };

            double dA_dd[3] = { 0.0, 0.0, 0.0 };
            double dB_dd[3] = { s[0], s[1], s[2] };
            double dC_dd[3] = { 2.0*t[0], 2.0*t[1], 2.0*t[2] };
            double dD_dd[3] = { 0.0, 0.0, 0.0 };
            double dE_dd[3] = { r0[0], r0[1], r0[2] };

            double du_da[3] = {0.0,0.0,0.0}, du_db[3] = {0.0,0.0,0.0}, du_dc[3] = {0.0,0.0,0.0}, du_dd[3] = {0.0,0.0,0.0};
            double dv_da[3] = {0.0,0.0,0.0}, dv_db[3] = {0.0,0.0,0.0}, dv_dc[3] = {0.0,0.0,0.0}, dv_dd[3] = {0.0,0.0,0.0};

            for (int d = 0; d < 3; ++d) {
                du_da[d] = du_dA * dA_da[d] + du_dB * dB_da[d] + du_dC * dC_da[d] + du_dD * dD_da[d] + du_dE * dE_da[d];
                du_db[d] = du_dA * dA_db[d] + du_dB * dB_db[d] + du_dC * dC_db[d] + du_dD * dD_db[d] + du_dE * dE_db[d];
                du_dc[d] = du_dA * dA_dc[d] + du_dB * dB_dc[d] + du_dC * dC_dc[d] + du_dD * dD_dc[d] + du_dE * dE_dc[d];
                du_dd[d] = du_dA * dA_dd[d] + du_dB * dB_dd[d] + du_dC * dC_dd[d] + du_dD * dD_dd[d] + du_dE * dE_dd[d];

                dv_da[d] = dv_dA * dA_da[d] + dv_dB * dB_da[d] + dv_dC * dC_da[d] + dv_dD * dD_da[d] + dv_dE * dE_da[d];
                dv_db[d] = dv_dA * dA_db[d] + dv_dB * dB_db[d] + dv_dC * dC_db[d] + dv_dD * dD_db[d] + dv_dE * dE_db[d];
                dv_dc[d] = dv_dA * dA_dc[d] + dv_dB * dB_dc[d] + dv_dC * dC_dc[d] + dv_dD * dD_dc[d] + dv_dE * dE_dc[d];
                dv_dd[d] = dv_dA * dA_dd[d] + dv_dB * dB_dd[d] + dv_dC * dC_dd[d] + dv_dD * dD_dd[d] + dv_dE * dE_dd[d];
            }

            // J^T r for each endpoint:
            double Jtr_a[3], Jtr_b[3], Jtr_c[3], Jtr_d[3];
            for (int d = 0; d < 3; ++d) {
                Jtr_a[d] = (1.0 - u) * rvec[d] + du_da[d] * dot_sr - dv_da[d] * dot_tr;
                Jtr_b[d] = (u)       * rvec[d] + du_db[d] * dot_sr - dv_db[d] * dot_tr;
                Jtr_c[d] = (v - 1.0) * rvec[d] + du_dc[d] * dot_sr - dv_dc[d] * dot_tr;
                Jtr_d[d] = (-v)      * rvec[d] + du_dd[d] * dot_sr - dv_dd[d] * dot_tr;
            }

            // gradient contribution = (dU/dd) * (1/d) * J^T r
            double scale = dU * (1.0 / dlen);
            for (int d = 0; d < 3; ++d) {
                addg(i,   d, scale * Jtr_a[d]);
                addg(i+1, d, scale * Jtr_b[d]);
                addg(j,   d, scale * Jtr_c[d]);
                addg(j+1, d, scale * Jtr_d[d]);
            }
        }
    }

    *energy_out = E;
}

} // extern "C"