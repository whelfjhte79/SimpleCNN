// SimpleCNN - CUDA 런타임
//
// 구조
//   DeviceApi   : GPU 에 실제로 일을 시키는 최소 인터페이스 (메모리, 복사, 커널 실행, GEMM)
//   DriverApi   : DeviceApi 구현. nvcuda / nvrtc / cublas 를 실행 중에 찾아서 연결한다.
//                 → 프로젝트에 CUDA 라이브러리를 링크하지 않아도 되고,
//                   GPU 가 없는 PC 에서도 같은 exe 가 CPU 로 그대로 돈다.
//   Runtime     : 프로그램 전체에서 하나. 메모리 풀(cudaMalloc 반복 방지)과 커널 실행 도우미.
#ifndef SIMPLE_CNN_CUDA_H
#define SIMPLE_CNN_CUDA_H

#include "Core.h"
#include "CudaKernels.h"

#include <filesystem>
#include <map>
#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define SCNN_CUDAAPI __stdcall
#else
#include <dlfcn.h>
#define SCNN_CUDAAPI
#endif

namespace cnn {
namespace cuda {

	class DeviceApi {
	public:
		virtual ~DeviceApi() {}
		virtual bool init(std::string& info, std::string& error) = 0;
		virtual bool loadKernels(const std::string& source, std::string& error) = 0;
		virtual void* alloc(size_t bytes) = 0;
		virtual void release(void* ptr) = 0;
		virtual void upload(void* dst, const void* src, size_t bytes) = 0;
		virtual void download(void* dst, const void* src, size_t bytes) = 0;
		virtual void copy(void* dst, const void* src, size_t bytes) = 0;
		virtual void launch(const char* kernel, unsigned gridX, unsigned gridY, unsigned block, void** params) = 0;
		virtual void synchronize() = 0;
		// 행 우선 GEMM. 빠른 라이브러리(cuBLAS)가 있으면 true, 없으면 false 를 돌려 커널로 처리
		virtual bool gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) { return false; }
		virtual size_t freeMemory() { return 0; }
	};

	// ---------------------------------------------------------------- 동적 라이브러리
	inline void* openLibrary(const std::string& path) {
#ifdef _WIN32
		if (path.find('\\') != std::string::npos || path.find('/') != std::string::npos)
			return (void*)LoadLibraryExA(path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
		return (void*)LoadLibraryA(path.c_str());
#else
		return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
	}
	inline void* findSymbol(void* library, const char* name) {
#ifdef _WIN32
		return (void*)GetProcAddress((HMODULE)library, name);
#else
		return dlsym(library, name);
#endif
	}

	// CUDA 툴킷이 있을 만한 폴더 (환경변수 → 기본 설치 경로, 최신 버전 우선)
	inline std::vector<std::string> toolkitDirs() {
		std::vector<std::string> dirs;
		const char* envs[] = { "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT" };
		for (const char* env : envs) {
			std::string root = envVar(env);
			if (root.empty())
				continue;
			dirs.push_back(root + "/bin");
			dirs.push_back(root + "/bin/x64");
			dirs.push_back(root + "/lib64");
			dirs.push_back(root + "/lib");
		}
		std::vector<std::string> roots;
#ifdef _WIN32
		roots.push_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
#else
		roots.push_back("/usr/local");
#endif
		for (const std::string& root : roots) {
			std::error_code ec;
			std::vector<std::string> found;
			for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
				std::string name = entry.path().filename().string();
				if (entry.is_directory(ec) && (name.rfind("v", 0) == 0 || name.rfind("cuda", 0) == 0))
					found.push_back(entry.path().string());
			}
			std::sort(found.rbegin(), found.rend());
			for (const std::string& dir : found) {
				dirs.push_back(dir + "/bin");
				dirs.push_back(dir + "/bin/x64");
				dirs.push_back(dir + "/lib64");
			}
		}
		return dirs;
	}

	// 폴더들에서 prefix 로 시작하는 라이브러리를 찾아 처음 열리는 것을 돌려준다.
	inline void* openFromToolkit(const std::string& prefix, const std::string& exclude, const std::vector<std::string>& plainNames, std::string& loadedName) {
		for (const std::string& name : plainNames) {
			void* lib = openLibrary(name);
			if (lib != nullptr) {
				loadedName = name;
				return lib;
			}
		}
		for (const std::string& dir : toolkitDirs()) {
			std::error_code ec;
			std::vector<std::string> files;
			for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
				std::string name = entry.path().filename().string();
				if (name.rfind(prefix, 0) != 0 || name.find(exclude) != std::string::npos)
					continue;
#ifdef _WIN32
				if (name.size() < 4 || name.substr(name.size() - 4) != ".dll")
					continue;
#else
				if (name.find(".so") == std::string::npos)
					continue;
#endif
				files.push_back(entry.path().string());
			}
			std::sort(files.rbegin(), files.rend());
			for (const std::string& file : files) {
				void* lib = openLibrary(file);
				if (lib != nullptr) {
					loadedName = file;
					return lib;
				}
			}
		}
		return nullptr;
	}

	// ---------------------------------------------------------------- 실제 GPU 구현
	class DriverApi : public DeviceApi {
	private:
		typedef int CUresult;
		typedef int CUdevice;
		typedef struct CUctx_st* CUcontext;
		typedef struct CUmod_st* CUmodule;
		typedef struct CUfunc_st* CUfunction;
		typedef struct CUstream_st* CUstream;
		typedef unsigned long long CUdeviceptr;
		typedef struct _nvrtcProgram* nvrtcProgram;
		typedef struct cublasContext* cublasHandle_t;

		CUresult(SCNN_CUDAAPI* cuInit)(unsigned int) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDriverGetVersion)(int*) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDeviceGetCount)(int*) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDeviceGet)(CUdevice*, int) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDeviceGetName)(char*, int, CUdevice) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDeviceTotalMem)(size_t*, CUdevice) = nullptr;
		CUresult(SCNN_CUDAAPI* cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
		CUresult(SCNN_CUDAAPI* cuCtxSetCurrent)(CUcontext) = nullptr;
		CUresult(SCNN_CUDAAPI* cuCtxSynchronize)() = nullptr;
		CUresult(SCNN_CUDAAPI* cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
		CUresult(SCNN_CUDAAPI* cuMemFree)(CUdeviceptr) = nullptr;
		CUresult(SCNN_CUDAAPI* cuMemGetInfo)(size_t*, size_t*) = nullptr;
		CUresult(SCNN_CUDAAPI* cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
		CUresult(SCNN_CUDAAPI* cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
		CUresult(SCNN_CUDAAPI* cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t) = nullptr;
		CUresult(SCNN_CUDAAPI* cuModuleLoadData)(CUmodule*, const void*) = nullptr;
		CUresult(SCNN_CUDAAPI* cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
		CUresult(SCNN_CUDAAPI* cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**) = nullptr;
		CUresult(SCNN_CUDAAPI* cuGetErrorString)(CUresult, const char**) = nullptr;

		int (*nvrtcVersion)(int*, int*) = nullptr;
		int (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
		int (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
		int (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
		int (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
		int (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;
		int (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;
		int (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
		int (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
		int (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
		int (*nvrtcGetNumSupportedArchs)(int*) = nullptr;
		int (*nvrtcGetSupportedArchs)(int*) = nullptr;

		int (SCNN_CUDAAPI* cublasCreate)(cublasHandle_t*) = nullptr;
		int (SCNN_CUDAAPI* cublasSgemm)(cublasHandle_t, int, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) = nullptr;

		void* driverLib = nullptr;
		void* nvrtcLib = nullptr;
		void* cublasLib = nullptr;
		CUdevice device = 0;
		CUcontext context = nullptr;
		CUmodule module = nullptr;
		cublasHandle_t cublas = nullptr;
		int computeCapability = 0;
		std::unordered_map<std::string, CUfunction> functions;

		template <typename T>
		bool bind(void* lib, T& fn, const char* name) {
			fn = reinterpret_cast<T>(findSymbol(lib, name));
			return fn != nullptr;
		}
		void check(CUresult result, const char* what) {
			if (result == 0)
				return;
			const char* message = "unknown";
			if (this->cuGetErrorString != nullptr)
				this->cuGetErrorString(result, &message);
			throw std::runtime_error(std::string("CUDA 오류 (") + what + "): " + message);
		}
		void makeCurrent() {
			this->cuCtxSetCurrent(this->context);
		}
		bool loadDriver(std::string& error) {
#ifdef _WIN32
			this->driverLib = openLibrary("nvcuda.dll");
#else
			this->driverLib = openLibrary("libcuda.so.1");
			if (this->driverLib == nullptr)
				this->driverLib = openLibrary("libcuda.so");
#endif
			if (this->driverLib == nullptr) {
				error = "NVIDIA 드라이버(nvcuda)를 찾지 못했습니다.";
				return false;
			}
			bool ok = true;
			ok = ok && bind(this->driverLib, this->cuInit, "cuInit");
			ok = ok && bind(this->driverLib, this->cuDriverGetVersion, "cuDriverGetVersion");
			ok = ok && bind(this->driverLib, this->cuDeviceGetCount, "cuDeviceGetCount");
			ok = ok && bind(this->driverLib, this->cuDeviceGet, "cuDeviceGet");
			ok = ok && bind(this->driverLib, this->cuDeviceGetName, "cuDeviceGetName");
			ok = ok && bind(this->driverLib, this->cuDeviceGetAttribute, "cuDeviceGetAttribute");
			ok = ok && bind(this->driverLib, this->cuDeviceTotalMem, "cuDeviceTotalMem_v2");
			ok = ok && bind(this->driverLib, this->cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
			ok = ok && bind(this->driverLib, this->cuCtxSetCurrent, "cuCtxSetCurrent");
			ok = ok && bind(this->driverLib, this->cuCtxSynchronize, "cuCtxSynchronize");
			ok = ok && bind(this->driverLib, this->cuMemAlloc, "cuMemAlloc_v2");
			ok = ok && bind(this->driverLib, this->cuMemFree, "cuMemFree_v2");
			ok = ok && bind(this->driverLib, this->cuMemGetInfo, "cuMemGetInfo_v2");
			ok = ok && bind(this->driverLib, this->cuMemcpyHtoD, "cuMemcpyHtoD_v2");
			ok = ok && bind(this->driverLib, this->cuMemcpyDtoH, "cuMemcpyDtoH_v2");
			ok = ok && bind(this->driverLib, this->cuMemcpyDtoD, "cuMemcpyDtoD_v2");
			ok = ok && bind(this->driverLib, this->cuModuleLoadData, "cuModuleLoadData");
			ok = ok && bind(this->driverLib, this->cuModuleGetFunction, "cuModuleGetFunction");
			ok = ok && bind(this->driverLib, this->cuLaunchKernel, "cuLaunchKernel");
			ok = ok && bind(this->driverLib, this->cuGetErrorString, "cuGetErrorString");
			if (!ok)
				error = "NVIDIA 드라이버가 너무 오래되었습니다. 드라이버를 업데이트하세요.";
			return ok;
		}
		bool loadNvrtc(std::string& error) {
			std::string name;
			std::vector<std::string> plain;
#ifdef _WIN32
			this->nvrtcLib = openFromToolkit("nvrtc64_", "builtins", plain, name);
#else
			plain = { "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2" };
			this->nvrtcLib = openFromToolkit("libnvrtc.so", "builtins", plain, name);
#endif
			if (this->nvrtcLib == nullptr) {
				error = "NVRTC(nvrtc64_*.dll)를 찾지 못했습니다. CUDA Toolkit 을 설치하거나 CUDA_PATH 환경변수를 확인하세요.";
				return false;
			}
			bool ok = true;
			ok = ok && bind(this->nvrtcLib, this->nvrtcVersion, "nvrtcVersion");
			ok = ok && bind(this->nvrtcLib, this->nvrtcCreateProgram, "nvrtcCreateProgram");
			ok = ok && bind(this->nvrtcLib, this->nvrtcCompileProgram, "nvrtcCompileProgram");
			ok = ok && bind(this->nvrtcLib, this->nvrtcGetPTXSize, "nvrtcGetPTXSize");
			ok = ok && bind(this->nvrtcLib, this->nvrtcGetPTX, "nvrtcGetPTX");
			ok = ok && bind(this->nvrtcLib, this->nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
			ok = ok && bind(this->nvrtcLib, this->nvrtcGetProgramLog, "nvrtcGetProgramLog");
			ok = ok && bind(this->nvrtcLib, this->nvrtcDestroyProgram, "nvrtcDestroyProgram");
			// 없어도 되는 함수 (구버전 NVRTC)
			bind(this->nvrtcLib, this->nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
			bind(this->nvrtcLib, this->nvrtcGetCUBIN, "nvrtcGetCUBIN");
			bind(this->nvrtcLib, this->nvrtcGetNumSupportedArchs, "nvrtcGetNumSupportedArchs");
			bind(this->nvrtcLib, this->nvrtcGetSupportedArchs, "nvrtcGetSupportedArchs");
			if (!ok)
				error = "NVRTC 라이브러리를 읽지 못했습니다: " + name;
			return ok;
		}
		void loadCublas() {
			if (!envVar("SIMPLECNN_NO_CUBLAS").empty())
				return;
			std::string name;
			std::vector<std::string> plain;
#ifdef _WIN32
			this->cublasLib = openFromToolkit("cublas64_", "cublasLt", plain, name);
#else
			plain = { "libcublas.so", "libcublas.so.13", "libcublas.so.12", "libcublas.so.11" };
			this->cublasLib = openFromToolkit("libcublas.so", "cublasLt", plain, name);
#endif
			if (this->cublasLib == nullptr)
				return;
			bool ok = true;
			ok = ok && bind(this->cublasLib, this->cublasCreate, "cublasCreate_v2");
			ok = ok && bind(this->cublasLib, this->cublasSgemm, "cublasSgemm_v2");
			if (!ok || this->cublasCreate(&this->cublas) != 0)
				this->cublas = nullptr;
		}
		// 이 GPU 에 맞는 아키텍처 선택. 정확히 지원하면 바로 실행 가능한 CUBIN, 아니면 PTX(JIT)
		void chooseArch(int& arch, bool& cubin) {
			arch = this->computeCapability;
			cubin = this->nvrtcGetCUBIN != nullptr;
			if (this->nvrtcGetNumSupportedArchs == nullptr || this->nvrtcGetSupportedArchs == nullptr)
				return;
			int count = 0;
			this->nvrtcGetNumSupportedArchs(&count);
			std::vector<int> archs(count);
			if (count > 0)
				this->nvrtcGetSupportedArchs(archs.data());
			int best = 0;
			bool exact = false;
			for (int a : archs) {
				if (a <= this->computeCapability && a > best)
					best = a;
				if (a == this->computeCapability)
					exact = true;
			}
			if (best > 0)
				arch = best;
			if (!exact)
				cubin = false;
		}
	public:
		bool init(std::string& info, std::string& error) override {
			if (!this->loadDriver(error))
				return false;
			if (this->cuInit(0) != 0) {
				error = "cuInit 실패: 사용할 수 있는 NVIDIA GPU 가 없습니다.";
				return false;
			}
			int count = 0;
			this->cuDeviceGetCount(&count);
			if (count <= 0) {
				error = "NVIDIA GPU 를 찾지 못했습니다.";
				return false;
			}
			int index = 0;
			std::string selected = envVar("SIMPLECNN_DEVICE");
			if (!selected.empty())
				index = (std::min)((std::max)(std::atoi(selected.c_str()), 0), count - 1);
			this->check(this->cuDeviceGet(&this->device, index), "cuDeviceGet");
			this->check(this->cuDevicePrimaryCtxRetain(&this->context, this->device), "cuDevicePrimaryCtxRetain");
			this->makeCurrent();

			char name[256] = { 0 };
			int major = 0;
			int minor = 0;
			int driverVersion = 0;
			size_t totalMem = 0;
			this->cuDeviceGetName(name, 255, this->device);
			this->cuDeviceGetAttribute(&major, 75, this->device);
			this->cuDeviceGetAttribute(&minor, 76, this->device);
			this->cuDriverGetVersion(&driverVersion);
			this->cuDeviceTotalMem(&totalMem, this->device);
			this->computeCapability = major * 10 + minor;

			if (!this->loadNvrtc(error))
				return false;
			this->loadCublas();

			std::ostringstream os;
			os << name << " (sm_" << this->computeCapability << ", " << formatBytes((double)totalMem)
				<< ", driver " << driverVersion / 1000 << "." << (driverVersion % 1000) / 10;
			if (this->cublas != nullptr)
				os << ", cuBLAS";
			os << ")";
			info = os.str();
			return true;
		}
		bool loadKernels(const std::string& source, std::string& error) override {
			int arch = 0;
			bool cubin = false;
			this->chooseArch(arch, cubin);
			std::string archOption = "--gpu-architecture=compute_" + std::to_string(arch);
			if (cubin)
				archOption = "--gpu-architecture=sm_" + std::to_string(arch);
			const char* options[] = { archOption.c_str(), "--use_fast_math" };

			nvrtcProgram program = nullptr;
			if (this->nvrtcCreateProgram(&program, source.c_str(), "simplecnn_kernels.cu", 0, nullptr, nullptr) != 0) {
				error = "nvrtcCreateProgram 실패";
				return false;
			}
			int result = this->nvrtcCompileProgram(program, 2, options);
			if (result != 0) {
				size_t logSize = 0;
				this->nvrtcGetProgramLogSize(program, &logSize);
				std::string log(logSize, '\0');
				this->nvrtcGetProgramLog(program, &log[0]);
				this->nvrtcDestroyProgram(&program);
				error = "CUDA 커널 컴파일 실패 (" + archOption + "):\n" + log;
				return false;
			}
			std::string image;
			size_t size = 0;
			if (cubin) {
				this->nvrtcGetCUBINSize(program, &size);
				image.resize(size);
				this->nvrtcGetCUBIN(program, &image[0]);
			}
			else {
				this->nvrtcGetPTXSize(program, &size);
				image.resize(size);
				this->nvrtcGetPTX(program, &image[0]);
			}
			this->nvrtcDestroyProgram(&program);

			this->makeCurrent();
			CUresult loaded = this->cuModuleLoadData(&this->module, image.data());
			if (loaded != 0) {
				error = "CUDA 커널 로드 실패. 그래픽 드라이버가 CUDA Toolkit 보다 오래된 경우입니다. 드라이버를 업데이트하세요.";
				return false;
			}
			for (const char* kernel : kKernelNames) {
				CUfunction fn = nullptr;
				this->check(this->cuModuleGetFunction(&fn, this->module, kernel), kernel);
				this->functions[kernel] = fn;
			}
			return true;
		}
		void* alloc(size_t bytes) override {
			this->makeCurrent();
			CUdeviceptr ptr = 0;
			if (this->cuMemAlloc(&ptr, bytes) != 0)
				return nullptr;
			return (void*)(uintptr_t)ptr;
		}
		void release(void* ptr) override {
			this->makeCurrent();
			this->cuMemFree((CUdeviceptr)(uintptr_t)ptr);
		}
		void upload(void* dst, const void* src, size_t bytes) override {
			this->makeCurrent();
			this->check(this->cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, bytes), "cuMemcpyHtoD");
		}
		void download(void* dst, const void* src, size_t bytes) override {
			this->makeCurrent();
			this->check(this->cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, bytes), "cuMemcpyDtoH");
		}
		void copy(void* dst, const void* src, size_t bytes) override {
			this->makeCurrent();
			this->check(this->cuMemcpyDtoD((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, bytes), "cuMemcpyDtoD");
		}
		void launch(const char* kernel, unsigned gridX, unsigned gridY, unsigned block, void** params) override {
			this->makeCurrent();
			CUfunction fn = this->functions.at(kernel);
			this->check(this->cuLaunchKernel(fn, gridX, gridY, 1, block, 1, 1, 0, nullptr, params, nullptr), kernel);
		}
		void synchronize() override {
			this->makeCurrent();
			this->check(this->cuCtxSynchronize(), "cuCtxSynchronize");
		}
		// cuBLAS 는 열 우선이므로 C^T = op(B)^T * op(A)^T 로 바꿔 부른다.
		bool gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) override {
			if (this->cublas == nullptr)
				return false;
			this->makeCurrent();
			int opA = 0;
			int opB = 0;
			if (transA)
				opA = 1;
			if (transB)
				opB = 1;
			int status = this->cublasSgemm(this->cublas, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
			if (status != 0)
				throw std::runtime_error("cublasSgemm 실패: " + std::to_string(status));
			return true;
		}
		size_t freeMemory() override {
			this->makeCurrent();
			size_t freeBytes = 0;
			size_t totalBytes = 0;
			this->cuMemGetInfo(&freeBytes, &totalBytes);
			return freeBytes;
		}
	};

	// ---------------------------------------------------------------- 런타임 (전역 하나)
	class Runtime {
	private:
		std::unique_ptr<DeviceApi> api;
		bool tried = false;
		bool ready = false;
		std::string infoText;
		std::string errorText;
		std::mutex mutex;
		// 메모리 풀: 크기별로 반납된 블록을 보관했다가 재사용한다.
		std::unordered_map<size_t, std::vector<void*>> pool;
		size_t usedBytes = 0;
		size_t cachedBytes = 0;
		size_t peakBytes = 0;

		Runtime() {}

		static size_t roundBytes(size_t bytes) {
			size_t unit = 4096;
			if (bytes >= ((size_t)1 << 20))
				unit = (size_t)1 << 20;
			return (bytes + unit - 1) / unit * unit;
		}
		void start() {
			this->tried = true;
			if (!this->api)
				this->api.reset(new DriverApi());
			try {
				if (!this->api->init(this->infoText, this->errorText))
					return;
				if (!this->api->loadKernels(kernelSource(), this->errorText))
					return;
				this->ready = true;
			}
			catch (const std::exception& e) {
				this->errorText = e.what();
			}
		}
	public:
		// NVRTC 로 컴파일할 전체 커널 소스
		static std::string kernelSource() {
			std::ostringstream os;
			os.setf(std::ios::fixed);
			os.precision(8);
			os << "#define SCNN_LEAKY " << kLeakySlope << "f\n";
			os << "#define SCNN_ELU " << kEluAlpha << "f\n";
			os << kKernelCommon << kKernelConv << kKernelMath;
			return os.str();
		}
		// 프로그램 종료 순서와 상관없이 안전하도록 일부러 해제하지 않는다.
		static Runtime& instance() {
			static Runtime* runtime = new Runtime();
			return *runtime;
		}
		// 다른 백엔드(테스트용 에뮬레이터 등)를 끼울 때 사용. 처음 사용하기 전에 호출해야 한다.
		void setApi(std::unique_ptr<DeviceApi> newApi) {
			this->api = std::move(newApi);
			this->tried = false;
			this->ready = false;
		}
		bool available() {
			std::lock_guard<std::mutex> lock(this->mutex);
			if (!this->tried)
				this->start();
			return this->ready;
		}
		const std::string& info() const { return this->infoText; }
		const std::string& error() const { return this->errorText; }
		size_t memoryInUse() const { return this->usedBytes; }
		size_t memoryPeak() const { return this->peakBytes; }
		size_t memoryCached() const { return this->cachedBytes; }

		void* alloc(size_t bytes) {
			std::lock_guard<std::mutex> lock(this->mutex);
			size_t rounded = roundBytes(bytes);
			std::vector<void*>& blocks = this->pool[rounded];
			void* ptr = nullptr;
			if (!blocks.empty()) {
				ptr = blocks.back();
				blocks.pop_back();
				this->cachedBytes -= rounded;
			}
			else {
				ptr = this->api->alloc(rounded);
				if (ptr == nullptr) {
					this->releaseCacheLocked();
					ptr = this->api->alloc(rounded);
				}
				if (ptr == nullptr)
					throw std::runtime_error("GPU 메모리 부족 (" + formatBytes((double)rounded) + " 요청). 배치 크기를 줄이거나 setWorkspaceLimit 를 낮추세요.");
			}
			this->usedBytes += rounded;
			this->peakBytes = (std::max)(this->peakBytes, this->usedBytes);
			return ptr;
		}
		void release(void* ptr, size_t bytes) {
			if (ptr == nullptr)
				return;
			std::lock_guard<std::mutex> lock(this->mutex);
			size_t rounded = roundBytes(bytes);
			this->pool[rounded].push_back(ptr);
			this->usedBytes -= rounded;
			this->cachedBytes += rounded;
		}
		void releaseCacheLocked() {
			for (auto& entry : this->pool) {
				for (void* ptr : entry.second)
					this->api->release(ptr);
				entry.second.clear();
			}
			this->cachedBytes = 0;
		}
		// 풀에 쌓아둔 블록을 실제로 GPU 에 돌려준다.
		void releaseCache() {
			std::lock_guard<std::mutex> lock(this->mutex);
			this->releaseCacheLocked();
		}
		void upload(void* dst, const void* src, size_t bytes) { this->api->upload(dst, src, bytes); }
		void download(void* dst, const void* src, size_t bytes) { this->api->download(dst, src, bytes); }
		void copy(void* dst, const void* src, size_t bytes) { this->api->copy(dst, src, bytes); }
		void synchronize() { this->api->synchronize(); }
		size_t freeMemory() { return this->api->freeMemory(); }

		// 1차원 그리드로 커널 실행. 커널은 모두 grid-stride 루프라서 블록 수에 상한을 둔다.
		template <typename... Args>
		void launch(const char* kernel, size_t total, Args... args) {
			size_t blocks = (total + 255) / 256;
			blocks = (std::max)((size_t)1, (std::min)(blocks, (size_t)65535));
			void* params[] = { (void*)&args... };
			this->api->launch(kernel, (unsigned)blocks, 1, 256, params);
		}
		template <typename... Args>
		void launchGrid(const char* kernel, unsigned gridX, unsigned gridY, unsigned block, Args... args) {
			void* params[] = { (void*)&args... };
			this->api->launch(kernel, gridX, gridY, block, params);
		}
		void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
			if (M <= 0 || N <= 0)
				return;
			if (this->api->gemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc))
				return;
			int ta = 0;
			int tb = 0;
			if (transA)
				ta = 1;
			if (transB)
				tb = 1;
			this->launchGrid("gemm_kernel", (unsigned)((N + 63) / 64), (unsigned)((M + 63) / 64), 256, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
		}
	};
}
}

#endif
