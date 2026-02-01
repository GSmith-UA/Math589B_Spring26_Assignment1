#include <cmath>
#include <algorithm>

extern "C" {

// Bump when you change the exported function signatures.
int rod_api_version() { return 2; }

double[2] computeClosest(i,j)
{
    double[2] solution = {0,0}

    std::vector<double> r = {get(i,0) - get(j,0), get(i,1) - get(j,1), get(i,2) - get(j,2)} 
    std::vector<double> d_i = {get(i+1,0) - get(i,0), get(i+1,1) - get(i,1), get(i+1,2) - get(i,2)}
    std::vector<double> d_j =  {get(j+1,0) - get(j,0), get(j+1,1) - get(j,1), get(j+1,2) - get(j,2)}

    // Coefficients for optimization
    double a = std::inner_product(d_i.begin(),d_i.end(),d_i.begin(),0.0)
    double b = std::inner_product(d_i.begin(),d_i.end(),d_j.begin(),0.0)
    double c = std::inner_product(d_j.begin(),d_j.end(),d_j.begin(),0.0)
    
    double d = 2*(std::inner_product(r.begin(),r.end(),d_i.begin(),0.0))
    double e = -2*(std::inner_product(r.begin(),r.end(),d_j.begin(),0.0))


    // Now we optimize using written HW algorithm
    // Compute the interior solution
    double det = a*c - b*b;
    double u_star, v_star;

    bool hasInterior = false;

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

    // Edge 1
    std::vector<std::pair<double,double>> candidates;

    if (c > 1e-12) {
        double v0 = std::clamp(-e / (2*c), 0.0, 1.0);
        candidates.emplace_back(0.0, v0);
    }

    // Edge 2
    if (c > 1e-12) {
        double v1 = std::clamp((2*b - e) / (2*c), 0.0, 1.0);
        candidates.emplace_back(1.0, v1);
    }

    // Edge 3
    if (a > 1e-12) {
        double u0 = std::clamp(-d / (2*a), 0.0, 1.0);
        candidates.emplace_back(u0, 0.0);
    }   

    // Edge 4
    if (a > 1e-12) {
        double u1 = std::clamp((2*b - d) / (2*a), 0.0, 1.0);
        candidates.emplace_back(u1, 1.0);
    }

    // Corners
    candidates.emplace_back(0.0, 0.0);
    candidates.emplace_back(0.0, 1.0);
    candidates.emplace_back(1.0, 0.0);
    candidates.emplace_back(1.0, 1.0);


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
}


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

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            // Rule out the two segments that are next
            int prevSeg = (i-1) % N;
            int nextSeg = (i+1) % N;
            if ((j == prevSeg) or (j == nextSeg))
            {
                continue 
            }
            optimal = computeClosest(i,j);

            double tempx = 0.0;
            double tempy = 0.0;
            double tempz = 0.0;

            tempx = (get(i,0) + optimal[0]*(get(i+1,0) - get(i,0))) - (get(j,0) + optimal[1]*(get(j+1,0) - get(j,0)))
            tempy = (get(i,1) + optimal[0]*(get(i+1,1) - get(i,1))) - (get(j,1) + optimal[1]*(get(j+1,1) - get(j,1))) 
            tempz = (get(i,2) + optimal[0]*(get(i+1,2) - get(i,2))) - (get(j,2) + optimal[1]*(get(j+1,2) - get(j,2)))

            d = sqrt((tempx)*(tempx) + (tempy)*(tempy) + (tempz)*(tempz));

            if (d < (std::pow(2,(1/6))*sigma))
            {
                E += (4 * eps * (std::pow((sigma/d),12) - std::pow((sigma/d),6) ) + eps);
                addg(i,0,);
                addg(i,1,);
                addg(i,2,);

                addg(i+1,0,);
                addg(i+1,1,);
                addg(i+1,2,);

                addg(j,0,);
                addg(j,1,);
                addg(j,2,);
            
                addg(j+1,0,;)
                addg(j+1,1,);
                addg(j+1,2,);

            }
        }
    }
    

    *energy_out = E;
}

} // extern "C"