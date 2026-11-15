#pragma once
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace OpenPhoton {

inline void logTrace(const std::string& msg) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    std::filesystem::path dir;
#ifdef _WIN32
    if (const wchar_t* up = _wgetenv(L"USERPROFILE")) {
        dir = std::filesystem::path(up) / L"Documents" / L"OpenPhoton";
    }
#endif
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::ofstream f(dir / "debug.log", std::ios::app);
    if (f.is_open()) {
        f << msg << "\n";
        f.flush();
    }
}

} // namespace OpenPhoton
