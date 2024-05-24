
#include <iostream>
#include <utility>
#include <vector>

#include <H5Cpp.h>

#include "fmt/format.h"
#include "fmt/printf.h"
#include "fmt/ranges.h"

#include "ImagePair.hpp"
#include "Track.hpp"

std::pair<std::vector<int>, std::vector<ImagePair>> load_h5(
    const std::string& path)
{
    using namespace H5;

    std::vector<ImagePair> pairs;

    H5File file(path, H5F_ACC_RDONLY);

    int n_pairs;
    file.openAttribute("n_pairs").read(PredType::NATIVE_INT, &n_pairs);

    for (int i = 0; i < n_pairs; i++) {
        ImagePair p;
        Group g = file.openGroup(fmt::format("pair_{}", i));

        // Read H
        DataSet H_dataset = g.openDataSet("H");
        H_dataset.read(p.H, PredType::NATIVE_DOUBLE);

        // Read src_pts
        DataSet src_pts_dataset = g.openDataSet("src_pts");
        hsize_t src_pts_dims[2];
        src_pts_dataset.getSpace().getSimpleExtentDims(src_pts_dims, NULL);
        p.src_pts.resize(src_pts_dims[0]);
        /*  NOTE: This is a bit fishy; we're assuming that a vector of arrays
            is stored contiguously in memory. It seems to work though. */
        src_pts_dataset.read(p.src_pts.data()->data(), PredType::NATIVE_DOUBLE);

        // Read dst_pts
        DataSet dst_pts_dataset = g.openDataSet("dst_pts");
        hsize_t dst_pts_dims[2];
        dst_pts_dataset.getSpace().getSimpleExtentDims(dst_pts_dims, NULL);
        p.dst_pts.resize(dst_pts_dims[0]);
        dst_pts_dataset.read(p.dst_pts.data()->data(), PredType::NATIVE_DOUBLE);

        // Read attributes i and j
        g.openAttribute("i").read(PredType::NATIVE_INT, &p.i);
        g.openAttribute("j").read(PredType::NATIVE_INT, &p.j);

        pairs.push_back(p);
    }

    std::vector<int> cam_indices;
    DataSet cam_indices_dataset = file.openDataSet("cam_indices");
    hsize_t cam_indices_dims[2];
    cam_indices_dataset.getSpace().getSimpleExtentDims(cam_indices_dims, NULL);
    cam_indices.resize(cam_indices_dims[0]);
    cam_indices_dataset.read(cam_indices.data(), PredType::NATIVE_INT);
    return {cam_indices, pairs};
}

std::pair<std::vector<int>, std::vector<Track>> load_tracks(
    const std::string& path)
{
    using namespace H5;

    std::vector<int> track_uids;
    std::vector<Track> tracks;

    H5File file(path, H5F_ACC_RDONLY);

    int n_tracks;
    file.openAttribute("n_tracks").read(PredType::NATIVE_INT, &n_tracks);

    DataSet track_uids_dataset = file.openDataSet("track_uids");
    track_uids.resize(static_cast<size_t>(n_tracks));
    track_uids_dataset.read(track_uids.data(), PredType::NATIVE_INT);

    for (int uid : track_uids) {
        Track t;
        t.uid = uid;
        Group g = file.openGroup(fmt::format("track_{}", uid));

        DataSet pts_dataset = g.openDataSet("pts");
        hsize_t src_pts_dims[2];
        pts_dataset.getSpace().getSimpleExtentDims(src_pts_dims, NULL);
        t.pts.resize(src_pts_dims[0]);
        pts_dataset.read(t.pts.data(), PredType::NATIVE_DOUBLE);

        g.openAttribute("start_frame_idx")
            .read(PredType::NATIVE_INT, &t.start_frame_idx);
        tracks.push_back(t);
    }

    return {track_uids, tracks};
}

void export_cam_params(const std::string& path,
                       std::map<int, std::array<double, 4>> cam_params)
{
    using namespace H5;

    H5File file(path, H5F_ACC_TRUNC);

    for (const auto& [cam_idx, params] : cam_params) {
        std::string name = fmt::format("cam_{}", cam_idx);

        hsize_t dims[1] = {4};
        DataSpace dataspace(1, dims);

        DataSet dataset =
            file.createDataSet(name, PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(params.data(), PredType::NATIVE_DOUBLE);

        dataset
            .createAttribute("cam_idx", PredType::NATIVE_INT,
                             DataSpace(H5S_SCALAR))
            .write(PredType::NATIVE_INT, &cam_idx);
    }
}
