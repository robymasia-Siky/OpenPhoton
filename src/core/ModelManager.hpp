#pragma once

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#endif

namespace OpenPhoton {

struct ModelDescriptor {
    int id;
    const char* displayName;
    const wchar_t* filename;
    const wchar_t* host;
    const wchar_t* path;
};

inline const ModelDescriptor kCatalogModels[3] = {
    {
        0,
        "HTDemucs 4-Stem (MIT)",
        L"htdemucs_fp16weights.onnx",
        L"huggingface.co",
        L"/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx"
    },
    {
        1,
        "HTDemucs 6-Stem (+Guitar/Piano, MIT)",
        L"htdemucs_6s_fp16weights.onnx",
        L"huggingface.co",
        L"/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx"
    },
    {
        2,
        "HTDemucs FT Vocals (MIT)",
        L"htdemucs_ft_vocals.onnx",
        L"huggingface.co",
        L"/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals.onnx"
    }
};

class ModelManager {
public:
    ~ModelManager() { cancelAndJoin(); }

    static std::filesystem::path getModelsDir() {
        std::filesystem::path base;
#ifdef _WIN32
        if (const wchar_t* up = _wgetenv(L"USERPROFILE")) {
            base = std::filesystem::path(up) / L"Documents" / L"OpenPhoton" / L"Models";
        }
#endif
        if (base.empty()) base = std::filesystem::current_path() / "Models";
        std::error_code ec;
        std::filesystem::create_directories(base, ec);
        return base;
    }

    static std::filesystem::path getModelPath(int catalogId) {
        if (catalogId < 0 || catalogId > 2) return {};
        return getModelsDir() / kCatalogModels[catalogId].filename;
    }

    static bool isModelInstalled(int catalogId) {
        auto p = getModelPath(catalogId);
        std::error_code ec;
        return !p.empty() && std::filesystem::exists(p, ec) && std::filesystem::file_size(p, ec) > 1000000;
    }

    void cancelAndJoin() {
        cancel_.store(true, std::memory_order_release);
        if (dlThread_.joinable()) dlThread_.join();
        downloading_.store(false, std::memory_order_release);
    }

    bool isDownloading() const noexcept { return downloading_.load(std::memory_order_acquire); }
    float progress() const noexcept { return progress_.load(std::memory_order_relaxed); }

    void startDownload(int catalogId, std::function<void(std::wstring)> onFinished) {
        if (catalogId < 0 || catalogId > 2) return;
        cancelAndJoin();

        cancel_.store(false, std::memory_order_release);
        progress_.store(0.0f, std::memory_order_relaxed);
        downloading_.store(true, std::memory_order_release);

        dlThread_ = std::thread([this, catalogId, cb = std::move(onFinished)]() {
            const auto& desc = kCatalogModels[catalogId];
            const auto finalPath = getModelPath(catalogId);
            const auto tmpPath = finalPath.wstring() + L".tmp";
            bool ok = false;

#ifdef _WIN32
            HINTERNET hSession = WinHttpOpen(L"OpenPhoton/0.4", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                             WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
            if (hSession) {
                HINTERNET hConnect = WinHttpConnect(hSession, desc.host, INTERNET_DEFAULT_HTTPS_PORT, 0);
                if (hConnect) {
                    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", desc.path, nullptr,
                                                            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                                            WINHTTP_FLAG_SECURE);
                    if (hRequest) {
                        DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
                        WinHttpSetOption(hRequest, WINHTTP_OPTION_REDIRECT_POLICY, &policy, sizeof(policy));

                        if (WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                               WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
                            WinHttpReceiveResponse(hRequest, nullptr)) {

                            DWORD statusCode = 0, sz = sizeof(statusCode);
                            WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                                WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &sz, WINHTTP_NO_HEADER_INDEX);

                            if (statusCode == 200) {
                                uint64_t totalBytes = 0;
                                wchar_t lenBuf[64]{};
                                DWORD lenSz = sizeof(lenBuf);
                                if (WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_CONTENT_LENGTH,
                                                        WINHTTP_HEADER_NAME_BY_INDEX, lenBuf, &lenSz, WINHTTP_NO_HEADER_INDEX)) {
                                    totalBytes = _wcstoui64(lenBuf, nullptr, 10);
                                }

                                std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
                                if (out.is_open()) {
                                    char buf[65536];
                                    uint64_t downloaded = 0;
                                    DWORD avail = 0;
                                    while (!cancel_.load(std::memory_order_acquire) &&
                                           WinHttpQueryDataAvailable(hRequest, &avail) && avail > 0) {
                                        DWORD toRead = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
                                        DWORD bytesRead = 0;
                                        if (!WinHttpReadData(hRequest, buf, toRead, &bytesRead) || bytesRead == 0) break;
                                        out.write(buf, bytesRead);
                                        downloaded += bytesRead;
                                        if (totalBytes > 0) {
                                            progress_.store(static_cast<float>(downloaded) / static_cast<float>(totalBytes),
                                                            std::memory_order_relaxed);
                                        }
                                    }
                                    out.close();
                                    if (!cancel_.load(std::memory_order_acquire) && downloaded > 1000000) {
                                        std::error_code ec;
                                        std::filesystem::remove(finalPath, ec);
                                        std::filesystem::rename(tmpPath, finalPath, ec);
                                        ok = !ec;
                                    }
                                }
                            }
                        }
                        WinHttpCloseHandle(hRequest);
                    }
                    WinHttpCloseHandle(hConnect);
                }
                WinHttpCloseHandle(hSession);
            }
#endif
            downloading_.store(false, std::memory_order_release);
            if (ok && cb) {
                progress_.store(1.0f, std::memory_order_relaxed);
                cb(finalPath.wstring());
            }
        });
    }

private:
    std::atomic<bool> downloading_{false};
    std::atomic<bool> cancel_{false};
    std::atomic<float> progress_{0.0f};
    std::thread dlThread_;
};

} // namespace OpenPhoton
