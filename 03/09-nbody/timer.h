#pragma once

#include <chrono>

namespace lesson_timer_detail {

inline std::chrono::steady_clock::time_point &start_time() {
    static std::chrono::steady_clock::time_point value =
        std::chrono::steady_clock::now();
    return value;
}

}  // namespace lesson_timer_detail

inline void StartTimer() {
    lesson_timer_detail::start_time() = std::chrono::steady_clock::now();
}

inline double GetTimer() {
    const auto elapsed = std::chrono::steady_clock::now() -
        lesson_timer_detail::start_time();
    return std::chrono::duration<double, std::milli>(elapsed).count();
}