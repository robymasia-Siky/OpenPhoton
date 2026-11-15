#include "DeepFilterBridge.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <excpt.h>
#endif

namespace OpenPhoton {

void* DeepFilterBridge::sehCreate(FnCreate fn, const char* path, float atten) {
#ifdef _WIN32
    __try { return fn(path, atten, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
#else
    return fn(path, atten, nullptr);
#endif
}

size_t DeepFilterBridge::sehFrameLen(FnFrameLen fn, void* st) {
#ifdef _WIN32
    __try { return fn(st); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
#else
    return fn(st);
#endif
}

float DeepFilterBridge::sehProcess(FnProcess fn, void* st, float* in, float* out) {
#ifdef _WIN32
    __try { return fn(st, in, out); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0.0f; }
#else
    return fn(st, in, out);
#endif
}

void DeepFilterBridge::sehFree(FnFree fn, void* st) {
#ifdef _WIN32
    __try { fn(st); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
#else
    fn(st);
#endif
}

} // namespace OpenPhoton
