
/*
ImagePair p = load_h5(DATA_DIR "p0-10.h5");

// Initialize with p.src/dst_pts[0], converted to Eigen Vector2
RelativeReprojError e {
    Eigen::Vector2d(p.src_pts[0][0], p.src_pts[0][1]),
    Eigen::Vector2d(p.dst_pts[0][0], p.dst_pts[0][1]),
};

double cam1[] = {-0.074634, 2.802281, 0.078858, 417};
double cam2[] = {-0.073303, 2.919203, 0.123331, 417};
double residuals[2];

e(cam1, cam2, residuals);

std::cout << residuals[0] << " " << residuals[1] << std::endl;
*/