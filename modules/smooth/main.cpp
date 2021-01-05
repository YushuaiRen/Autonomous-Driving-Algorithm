#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

#include "matplotlibcpp.h"

#include "osqp/osqp.h"
#include "modules/common/include/vec2d.h"
#include "modules/smooth/include/fem_pos_deviation.h"
#include "modules/smooth/include/fem_pos_deviation_sqp_osqp_interface.h"

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
    const double lateral_bound = 0.25;
    std::vector<double> bounds(num, lateral_bound);

    bounds.front() = 0.0;
    bounds.back() = 0.0;


    raw_point2d[0].first = px[0];
    raw_point2d[0].second = std::sin(px[0]);
    randomx[0] = px[0];
    randomy[0] = std::sin(px[0]);
    sinx[0] = px[0];
    siny[0] = std::sin(px[0]);

    raw_point2d[num-1].first = px[num-1];
    raw_point2d[num-1].second = std::sin(px[num-1]);
    randomx[num-1] = px[num-1];
    randomy[num-1] = std::sin(px[num-1]);
    sinx[num-1] = px[num-1];
    siny[num-1] = std::sin(px[num-1]);

    std::random_device rd;
    std::default_random_engine gen = std::default_random_engine(rd());
    std::normal_distribution<> dis{0, 0.05};
    
    for (int i = 1; i < num - 1; ++i) {
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

    FemPosDeviationSqpOsqpInterface solver;
    solver.set_ref_points(raw_point2d);
    solver.set_bounds_around_refs(bounds);
  
    solver.Solve();
      
  
    std::vector<std::pair<double, double>> opt_xy = solver.opt_xy();
  
    // TODO(Jinyun): unify output data container
    opt_x.resize(opt_xy.size());
    opt_y.resize(opt_xy.size());
    for (size_t i = 0; i < opt_xy.size(); ++i) {
      opt_x[i] = opt_xy[i].first;
      opt_y[i] = opt_xy[i].second;
    }

    plt::plot(opt_x, opt_y);

    plt::grid(true);
    plt::show();


    return 0;
};