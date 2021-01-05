#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

#include "matplotlibcpp.h"

#include "osqp/osqp.h"
#include "modules/common/include/vec2d.h"
#include "modules/smooth/include/fem_pos_deviation.h"

using namespace planning;
namespace plt = matplotlibcpp;

int main(int argc, char **argv) {

    const int num = 100;
    const double low = 0;
    const double high = 2 * M_PI;

    Eigen::VectorXd px = Eigen::VectorXd::LinSpaced(num, low, high);
    
    std::vector<double> sinx(num, 0);
    std::vector<double> siny(num, 0);
    std::vector<double> randomx(num, 0);
    std::vector<double> randomy(num, 0);

    std::vector<std::pair<double, double>> raw_point2d(num, {0.0, 0.0});
    const double lateral_bound = 0.1;
    std::vector<double> bounds(num, lateral_bound);

    bounds.front() = 0.0;
    bounds.back() = 0.0;

    std::random_device rd;
    std::default_random_engine gen = std::default_random_engine(rd());
    std::normal_distribution<> dis{0, 0.05};
    for (int i = 0; i < num; ++i) {
        raw_point2d[i].first = px[i];
        raw_point2d[i].second = std::sin(px[i]) + dis(gen);
        randomx[i] = px[i];
        randomy[i] = std::sin(px[i]) + dis(gen);
        sinx[i] = px[i];
        siny[i] = std::sin(px[i]);
    }
    
    FemPosDeviation test;
    test.set_ref_points(raw_point2d);
    test.set_bounds_around_refs(bounds);

    test.Solve();
    
    std::vector<double> opt_x = test.opt_x();
    std::vector<double> opt_y = test.opt_y();

    plt::plot(sinx, siny);
    plt::plot(randomx, randomy);
    plt::plot(opt_x, opt_y);
    plt::grid(true);
    plt::show();


    return 0;
};