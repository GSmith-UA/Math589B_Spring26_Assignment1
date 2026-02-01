#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <numeric>
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
        std::array<double,2> solution{0.0, 0.0};

        std::array<double,3> r = {get(i,0) - get(j,0), get(i,1) - get(j,1), get(i,2) - get(j,2)};
        std::array<double,3> d_i = {get(i+1,0) - get(i,0), get(i+1,1) - get(i,1), get(i+1,2) - get(i,2)};
        std::array<double,3> d_j =  {get(j+1,0) - get(j,0), get(j+1,1) - get(j,1), get(j+1,2) - get(j,2)};

        // Coefficients for optimization
        double a = std::inner_product(d_i.begin(),d_i.end(),d_i.begin(),0.0);
        double b = std::inner_product(d_i.begin(),d_i.end(),d_j.begin(),0.0);
        double c = std::inner_product(d_j.begin(),d_j.end(),d_j.begin(),0.0);
        
        double d = 2*(std::inner_product(r.begin(),r.end(),d_i.begin(),0.0));
        double e = -2*(std::inner_product(r.begin(),r.end(),d_j.begin(),0.0));


        // Now we optimize using written HW algorithm
        // Compute the interior solution
        double det = a*c - b*b;
        double u_star, v_star;

        if (det > 1e-12) {  // non-parallel, numerically safe
            double r_di = std::inner_product(r.begin(), r.end(), d_i.begin(), 0.0);
            double r_dj = std::inner_product(r.begin(), r.end(), d_j.begin(), 0.0);

            u_star = ( b*r_dj - c*r_di ) / det;
            v_star = ( a*r_dj - b*r_di ) / det;

            if (u_star >= 0.0 && u_star <= 1.0 &&
                v_star >= 0.0 && v_star <= 1.0)
            {
                solution[0] = u_star;
                solution[1] = v_star;
                return solution;
            }
        }

        // Helper
        auto eval = [&](double u, double v) {
        double val = 0.0;
        for (int k = 0; k < 3; ++k) {
            double diff = r[k] + u*d_i[k] - v*d_j[k];
            val += diff * diff;
        }
        return val;
        };

        std::array<std::pair<double,double>,8> candidates;
        int count = 0;

        // Edge 1
        if (c > 1e-12) {
            double v0 = std::clamp(-e / (2*c), 0.0, 1.0);
            candidates[count++] = {0.0, v0};
        }

        // Edge 2
        if (c > 1e-12) {
            double v1 = std::clamp((2*b - e) / (2*c), 0.0, 1.0);
            candidates[count++] = {1.0, v1};
        }

        // Edge 3
        if (a > 1e-12) {
            double u0 = std::clamp(-d / (2*a), 0.0, 1.0);
            candidates[count++] = {u0, 0.0};
        }   

        // Edge 4
        if (a > 1e-12) {
            double u1 = std::clamp((2*b - d) / (2*a), 0.0, 1.0);
            candidates[count++] = {u1, 1.0};
        }

        // Corners
        candidates[count++] = {0.0,0.0};
        candidates[count++] = {0.0,1.0};
        candidates[count++] = {1.0,0.0};
        candidates[count++] = {1.0,1.0};


        // Find the solution
        double bestVal = std::numeric_limits<double>::infinity();

        for (auto& uv : candidates) {
            double val = eval(uv.first, uv.second);
            if (val < bestVal) {
                bestVal = val;
                solution[0] = uv.first;
                solution[1] = uv.second;
            }
        }
        return solution;
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

    for (int i = 0; i < N; ++i) 
    {
        for (int j = i+2; j < N; ++j) 
        {

            if ((i == 0 && j == N-1)) continue; // periodic neighbor

            auto optimal = computeClosest(i,j);
            double u = optimal[0];
            double v = optimal[1];
            double dist;
            double tempx = 0.0;
            double tempy = 0.0;
            double tempz = 0.0;

            tempx = (get(i,0) + optimal[0]*(get(i+1,0) - get(i,0))) - (get(j,0) + optimal[1]*(get(j+1,0) - get(j,0)));
            tempy = (get(i,1) + optimal[0]*(get(i+1,1) - get(i,1))) - (get(j,1) + optimal[1]*(get(j+1,1) - get(j,1))); 
            tempz = (get(i,2) + optimal[0]*(get(i+1,2) - get(i,2))) - (get(j,2) + optimal[1]*(get(j+1,2) - get(j,2)));

            dist = std::sqrt((tempx)*(tempx) + (tempy)*(tempy) + (tempz)*(tempz));
            dist = std::max(dist, 1e-12);
            const double cutoff = std::pow(2.0, 1.0/6.0) * sigma;
            if (dist < cutoff)
            {
                double invd = 1.0 / dist;
                double s2 = (sigma * invd) * (sigma * invd);
                double s6 = s2 * s2 * s2;

                E += 4 * eps * (s6*s6 - s6) + eps;
                double forceMag = 24 * eps * invd * (2*s6*s6 - s6);


                double fx = forceMag * tempx * invd;
                double fy = forceMag * tempy * invd;
                double fz = forceMag * tempz * invd;


                addg(i,   0,  (1-u)*fx);
                addg(i,   1,  (1-u)*fy);
                addg(i,   2,  (1-u)*fz);

                addg(i+1, 0,  u*fx);
                addg(i+1, 1,  u*fy);
                addg(i+1, 2,  u*fz);

                addg(j,   0, -(1-v)*fx);
                addg(j,   1, -(1-v)*fy);
                addg(j,   2, -(1-v)*fz);

                addg(j+1, 0, -v*fx);
                addg(j+1, 1, -v*fy);
                addg(j+1, 2, -v*fz);


            }
        }
    }
    

    *energy_out = E;
}

} // extern "C"