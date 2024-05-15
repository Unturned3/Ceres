
#pragma once

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

typedef unsigned int uint;

#ifdef LOG_FILE_AND_LINE
#define dlog(...) utils::log(__FILE__, __LINE__, __VA_ARGS__)
#else
#define dlog(...) utils::log(__VA_ARGS__)
#endif

#define check(e, m)                        \
    do {                                   \
        if (!(e)) {                        \
            throw std::runtime_error((m)); \
        }                                  \
    } while (0)

namespace utils {

// Variadic Templates
// https://stackoverflow.com/a/29326784
template <typename... Args>
void log(Args &&...args)
{
    (std::cout << ... << args) << std::endl;
}

template <typename T>
std::string pretty_matrix(const T *a, int n, int m, int sig_figs,
                          bool col_major = true)
{
    std::stringstream s;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int k = col_major ? (j * m + i) : (i * m + j);
            s << std::fixed << std::setprecision(sig_figs)
              << std::setw(sig_figs + 4) << *(a + k);
        }
    }
    return s.str();
}

// template <typename T> std::string type_name();

template <typename T>
class Timer {
public:
    Timer(std::string message, std::string unit)
        : start {std::chrono::high_resolution_clock::now()},
          message {"Elapsed time: "},
          unit {unit}
    {
    }

    Timer(std::string unit) : Timer("Elapsed time: ", unit) {}

    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<T>(end - start).count();
        std::cout << message << elapsed << " " << unit << std::endl;
    }

    Timer(const Timer &) = delete;
    Timer &operator=(const Timer &) = delete;

    Timer(Timer &&) = delete;
    Timer &operator=(Timer &&) = delete;

private:
    std::chrono::high_resolution_clock::time_point start;
    std::string message;
    std::string unit;
};

}  // namespace utils
