#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace py = pybind11;
using Eigen::Matrix3d;
using Eigen::Map;
using std::vector;

// Convert numpy array to Eigen 3x3 matrix
Matrix3d numpy_to_eigen(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.size != 9) {
        throw std::runtime_error("Expected 3x3 matrix");
    }
    double* ptr = static_cast<double*>(buf.ptr);
    return Map<Matrix3d>(ptr);
}

// Convert Eigen 3x3 matrix to numpy array
py::array_t<double> eigen_to_numpy(const Matrix3d& mat) {
    auto result = py::array_t<double>({3, 3});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    Map<Matrix3d>(ptr) = mat;
    return result;
}

// Smooth homographies using matrix log/exp averaging
std::vector<py::object> smooth_homographies(
    const std::vector<py::object>& Hs, 
    const std::vector<py::object>& prev_logs, 
    double alpha
) {
    std::vector<py::object> result;
    for (size_t i = 0; i < Hs.size(); ++i) {
        if (Hs[i].is_none()) {
            result.push_back(py::none());
            continue;
        }

        Matrix3d H = numpy_to_eigen(Hs[i].cast<py::array_t<double>>());
        Matrix3d L = H.log();

        if (prev_logs[i].is_none()) {
            result.push_back(eigen_to_numpy(H));
        } else {
            Matrix3d L_prev = numpy_to_eigen(prev_logs[i].cast<py::array_t<double>>());
            Matrix3d L_smooth = alpha * L + (1.0 - alpha) * L_prev;
            Matrix3d H_smooth = L_smooth.exp();
            H_smooth /= H_smooth(2, 2); // normalize scale
            result.push_back(eigen_to_numpy(H_smooth));
        }
    }

    return result;
}

PYBIND11_MODULE(utils, m) {
    m.def("smooth_homographies", &smooth_homographies,
          "Smooth batch of homographies using exponential averaging in Lie algebra",
          py::arg("Hs"), py::arg("prev_logs"), py::arg("alpha"));
}
