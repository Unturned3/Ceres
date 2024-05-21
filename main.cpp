
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>

#include "fmt/format.h"
#include "fmt/printf.h"
#include "fmt/ranges.h"

#include "cnpy.h"

#include "ImagePair.hpp"
#include "LoadH5.hpp"
#include "config.h"
#include "utils.hpp"

/*  Computes the relative reprojection error between two images observed
    by two cameras. We "unproject" a 2D point seen by cam1 and then project
    this point onto the image plane of cam2. The error is the difference
    between the reprojection and the observed 2D points in cam2.

    The cameras are parameterized by an axis-angle rotation and a focal length.
*/
struct RelativeReprojError {
    RelativeReprojError(const Eigen::Vector2d& src_pt,
                        const Eigen::Vector2d& observed_pt)
        : src_pt(src_pt), observed_pt(observed_pt)
    {
    }

    template <typename T>
    bool operator()(const T* const cam0, const T* const cam1,
                    T* residuals) const
    {
        /*  cam[0:3] is the axis-angle rotation, cam[3] is the focal length.
            Intrinsic matrix K of a camera has the form:
            [-f 0 320]
            [ 0 f 240]
            [ 0 0   1]
        */

        Eigen::Matrix<T, 3, 3> R0, R1;
        ceres::AngleAxisToRotationMatrix(cam0, R0.data());
        ceres::AngleAxisToRotationMatrix(cam1, R1.data());

        T f0 = cam0[3];
        T f1 = cam1[3];

        T cx = T(320);
        T cy = T(240);

        Eigen::Vector3<T> src_pt_3d {
            (src_pt[0] - cx) / -f0,
            (src_pt[1] - cy) / f0,
            T(1),
        };

        Eigen::Vector3<T> dst_pt_3d = R1.transpose() * R0 * src_pt_3d;

        Eigen::Vector2<T> dst_pt {
            (-f1 * dst_pt_3d[0]) / dst_pt_3d[2] + cx,
            (f1 * dst_pt_3d[1]) / dst_pt_3d[2] + cy,
        };

        residuals[0] = dst_pt[0] - T(observed_pt[0]);
        residuals[1] = dst_pt[1] - T(observed_pt[1]);

        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector2d& src_pt,
                                       const Eigen::Vector2d& observed_pt)
    {
        return new ceres::AutoDiffCostFunction<RelativeReprojError, 2, 4, 4>(
            new RelativeReprojError(src_pt, observed_pt));
    }

    static ceres::CostFunction* create(const std::array<double, 2> src_pt,
                                       const std::array<double, 2> observed_pt)
    {
        return new ceres::AutoDiffCostFunction<RelativeReprojError, 2, 4, 4>(
            new RelativeReprojError(
                Eigen::Vector2d(src_pt[0], src_pt[1]),
                Eigen::Vector2d(observed_pt[0], observed_pt[1])));
    }

private:
    const Eigen::Vector2d src_pt;
    const Eigen::Vector2d observed_pt;
};

int main()
{
    //cnpy::NpyArray gt_poses = cnpy::npy_load(DATA_DIR "000.npy");

    auto [cam_indices, image_pairs] = load_h5(DATA_DIR "cart10.h5");

    std::map<int, std::array<double, 4>> cam_params;
    for (int& i : cam_indices) {
        cam_params[i] = {0, 0, 0, 640};
    }

    ceres::Problem problem;
    for (const ImagePair& p : image_pairs) {
        for (size_t i = 0; i < p.src_pts.size(); i++) {
            ceres::CostFunction* cost_function =
                RelativeReprojError::create(p.src_pts[i], p.dst_pts[i]);
            problem.AddResidualBlock(cost_function, nullptr,
                                     cam_params[p.i].data(),
                                     cam_params[p.j].data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    fmt::print("Optimized camera params:\n");

    for (auto& v : cam_params) {
        auto [cam_id, params] = v;
        fmt::print("cam{}: {}\n", cam_id, fmt::join(params, ", "));
    }

    double final_cost = summary.final_cost;
    final_cost /= static_cast<double>(summary.num_residual_blocks);
    final_cost = std::sqrt(final_cost);
    fmt::print("\nAverage reprojection error: {:.2f} pixels\n", final_cost);

    export_cam_params(DATA_DIR "optimized_cam_params.h5", cam_params);
    return 0;
}
