
#include <tuple>
#include <vector>

#include <H5Cpp.h>
#include <H5PredType.h>
#include <Eigen/Dense>

#include "fmt/format.h"

namespace MyH5 {

std::vector<hsize_t> getDataSetDims(H5::DataSet& ds)
{
    auto ndims = ds.getSpace().getSimpleExtentNdims();
    std::vector<hsize_t> dims(ndims);
    ds.getSpace().getSimpleExtentDims(dims.data(), nullptr);
    return dims;
}

}  // namespace MyH5

std::tuple<int, int, std::vector<std::vector<Eigen::Vector2d>>>
load_sift_tracks(const std::string& path)
{
    using namespace H5;
    H5File f(path, H5F_ACC_RDONLY);
    std::vector<std::vector<Eigen::Vector2d>> tracks;

    int f_begin;
    f.openAttribute("f_begin").read(PredType::NATIVE_INT, &f_begin);

    auto g = f.openGroup(fmt::format("keyframe_{}", f_begin));

    int n_pts;
    g.openAttribute("n_pts").read(PredType::NATIVE_INT, &n_pts);

    fmt::print("f_begin: {}\n", f_begin);
    fmt::print("n_pts: {}\n", n_pts);

    tracks.resize(n_pts);

    for (int i = 0; i < n_pts; i++) {
        auto ds = g.openDataSet(fmt::format("pt_{}", i));
        auto dims = MyH5::getDataSetDims(ds);
        tracks[i].resize(dims[0]);
        ds.read(tracks[i][0].data(), PredType::NATIVE_DOUBLE);
    }

    return {f_begin, f_begin + tracks[0].size(), tracks};
}
