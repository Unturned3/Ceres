
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct RelativeReprojErrorWithMotion {
    RelativeReprojErrorWithMotion(const Eigen::Vector2d& src_pt,
                                  const Eigen::Vector2d& observed_pt,
                                  const double reg)
        : src_pt(src_pt), observed_pt(observed_pt), reg(reg)
    {
    }

    template <typename T>
    bool operator()(const T* const cam0, const T* const cam1,
                    const T* const pt_motion, T* residuals) const
    {
        /*  cam[0:3] is the axis-angle rotation, cam[3] is the focal length.
            Intrinsic matrix K of a camera has the form:
            [-f 0 320]
            [ 0 f 240]
            [ 0 0   1]
        */

        Eigen::Matrix<T, 3, 3> R0, R1, Rpt;
        ceres::AngleAxisToRotationMatrix(cam0, R0.data());
        ceres::AngleAxisToRotationMatrix(cam1, R1.data());
        ceres::AngleAxisToRotationMatrix(pt_motion, Rpt.data());

        T f0 = cam0[3];
        T f1 = cam1[3];

        T cx = T(320);
        T cy = T(240);

        Eigen::Vector3<T> src_pt_3d {
            (src_pt[0] - cx) / -f0,
            (src_pt[1] - cy) / f0,
            T(1),
        };

        Eigen::Vector3<T> dst_pt_3d = R1.transpose() * Rpt * R0 * src_pt_3d;

        Eigen::Vector2<T> dst_pt {
            (-f1 * dst_pt_3d[0]) / dst_pt_3d[2] + cx,
            (f1 * dst_pt_3d[1]) / dst_pt_3d[2] + cy,
        };

        residuals[0] = dst_pt[0] - T(observed_pt[0]);
        residuals[1] = dst_pt[1] - T(observed_pt[1]);

        // Regularize magnitude of pt_motion vector
        residuals[2] = pt_motion[0] * T(reg);
        residuals[3] = pt_motion[1] * T(reg);
        residuals[4] = pt_motion[2] * T(reg);
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector2d& src_pt,
                                       const Eigen::Vector2d& observed_pt,
                                       double reg)
    {
        return new ceres::AutoDiffCostFunction<RelativeReprojErrorWithMotion, 5,
                                               4, 4, 3>(
            new RelativeReprojErrorWithMotion(src_pt, observed_pt, reg));
    }

    static ceres::CostFunction* create(const std::array<double, 2> src_pt,
                                       const std::array<double, 2> observed_pt,
                                       double reg)
    {
        return new ceres::AutoDiffCostFunction<RelativeReprojErrorWithMotion, 5,
                                               4, 4, 3>(
            new RelativeReprojErrorWithMotion(
                Eigen::Vector2d(src_pt[0], src_pt[1]),
                Eigen::Vector2d(observed_pt[0], observed_pt[1]),
                reg));
    }

private:
    const Eigen::Vector2d src_pt;
    const Eigen::Vector2d observed_pt;
    double reg;
};
