
#include <cstddef>
#include <iostream>
#include <map>
#include <opencv2/videoio.hpp>
#include <vector>
#include <filesystem>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "cnpy.h"

#include "ImagePair.hpp"
#include "LoadH5.hpp"
#include "ReprojError.hpp"
#include "h5_rw.hpp"

namespace fs = std::filesystem;


int main(int argc, char** argv)
{
    if (argc < 3) {
        fmt::print("Usage: {} data_dir filename\n", argv[0]);
        return 1;
    }

    fs::path data_dir = argv[1];
    fs::path filename = argv[2];

    auto [cam_indices, image_pairs] = load_h5(data_dir / filename);

    fmt::print("Loaded {} image pairs.\n", image_pairs.size());

    std::map<int, std::array<double, 4>> cam_params;
    for (int& i : cam_indices) {
        cam_params[i] = {0, 0, 0, 640};
    }

    ceres::Problem problem;

    int scaled_cnt = 0;
    for (const ImagePair& p : image_pairs) {
        for (size_t i = 0; i < p.src_pts.size(); i++) {
            ceres::CostFunction* cost_function =
                RelativeReprojError::create(p.src_pts[i], p.dst_pts[i]);

            if (p.i != p.j - 1) {
                scaled_cnt += 1;
                problem.AddResidualBlock(
                    cost_function,
                    //new ceres::ScaledLoss(nullptr, 1000, ceres::TAKE_OWNERSHIP),
                    //new ceres::HuberLoss(100),
                    nullptr,
                    cam_params[p.i].data(),
                    cam_params[p.j].data()
                );
            } else {
                problem.AddResidualBlock(cost_function,
                                         new ceres::CauchyLoss(0.5),
                                         //new ceres::HuberLoss(2.0),
                                         //nullptr,
                                         cam_params[p.i].data(),
                                         cam_params[p.j].data());
            }

        }
    }

    fmt::print("Number of scaled residuals: {}\n", scaled_cnt);

    ceres::Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    //fmt::print("Optimized camera params:\n");
    //for (auto& v : cam_params) {
    //    auto [cam_id, params] = v;
    //    fmt::print("cam{}: {}\n", cam_id, fmt::join(params, ", "));
    //}

    double final_cost = summary.final_cost;
    final_cost /= static_cast<double>(summary.num_residual_blocks);
    final_cost = std::sqrt(final_cost);
    fmt::print("\nAverage reprojection error: {:.2f} pixels\n", final_cost);

    export_cam_params(data_dir / "optimized_cam_params.h5", cam_params);
    return 0;
}
