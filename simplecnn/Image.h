// SimpleCNN - 이미지 읽기/쓰기
// JPG, PNG, BMP, TGA, GIF, PSD, PNM 를 읽는다 (stb_image, 퍼블릭 도메인 단일 헤더).
// OpenCV 같은 외부 라이브러리 없이 어디서나 컴파일된다.
#ifndef SIMPLE_CNN_IMAGE_H
#define SIMPLE_CNN_IMAGE_H

#include "Core.h"

#include <cctype>
#include <filesystem>

#ifndef SIMPLECNN_NO_STB
#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "third_party/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
#endif

namespace cnn {

	// 경로 문자열은 OS 기본 인코딩 그대로 쓴다. (Windows 는 CP949 라서 한글 폴더명도 그대로 동작)
	inline std::filesystem::path toPath(const std::string& text) {
		return std::filesystem::path(text);
	}
	inline std::string pathString(const std::filesystem::path& path) {
		return path.string();
	}
	inline bool readFile(const std::filesystem::path& path, std::vector<uint8_t>& out) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file)
			return false;
		std::streamsize size = file.tellg();
		file.seekg(0);
		out.resize((size_t)size);
		return (bool)file.read(reinterpret_cast<char*>(out.data()), size);
	}

	struct Image {
		int width = 0;
		int height = 0;
		int channels = 0;
		std::vector<uint8_t> pixels; // HWC, RGB 순서

		Image() {}
		Image(int width, int height, int channels) : width(width), height(height), channels(channels), pixels((size_t)width * height * channels, 0) {}

		bool empty() const { return this->pixels.empty(); }
		uint8_t& at(int x, int y, int c) { return this->pixels[((size_t)y * this->width + x) * this->channels + c]; }
		uint8_t at(int x, int y, int c) const { return this->pixels[((size_t)y * this->width + x) * this->channels + c]; }

		static Image load(const std::filesystem::path& path) {
			Image image;
			std::vector<uint8_t> bytes;
			if (!readFile(path, bytes) || bytes.empty())
				return image;
#ifndef SIMPLECNN_NO_STB
			int w = 0;
			int h = 0;
			int c = 0;
			unsigned char* data = stbi_load_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &c, 0);
			if (data == nullptr)
				return image;
			image.width = w;
			image.height = h;
			image.channels = c;
			image.pixels.assign(data, data + (size_t)w * h * c);
			stbi_image_free(data);
#else
			image = decodeBMP(bytes);
#endif
			return image;
		}
		static Image load(const std::string& path) { return load(toPath(path)); }

		// 24비트 BMP 저장 (테스트 결과를 눈으로 확인할 때)
		bool saveBMP(const std::string& path) const {
			Image rgb = this->withChannels(3);
			int rowBytes = (rgb.width * 3 + 3) / 4 * 4;
			uint32_t dataSize = (uint32_t)rowBytes * rgb.height;
			uint8_t header[54] = { 0 };
			header[0] = 'B';
			header[1] = 'M';
			writeLE(header + 2, 54 + dataSize);
			writeLE(header + 10, 54);
			writeLE(header + 14, 40);
			writeLE(header + 18, (uint32_t)rgb.width);
			writeLE(header + 22, (uint32_t)rgb.height);
			header[26] = 1;
			header[28] = 24;
			writeLE(header + 34, dataSize);
			std::ofstream file(toPath(path), std::ios::binary);
			if (!file)
				return false;
			file.write(reinterpret_cast<const char*>(header), 54);
			std::vector<uint8_t> row(rowBytes, 0);
			for (int y = rgb.height - 1; y >= 0; --y) {
				for (int x = 0; x < rgb.width; ++x) {
					row[x * 3 + 0] = rgb.at(x, y, 2);
					row[x * 3 + 1] = rgb.at(x, y, 1);
					row[x * 3 + 2] = rgb.at(x, y, 0);
				}
				file.write(reinterpret_cast<const char*>(row.data()), rowBytes);
			}
			return (bool)file;
		}

		// 양선형 보간 크기 변환
		Image resized(int newWidth, int newHeight) const {
			if (newWidth == this->width && newHeight == this->height)
				return *this;
			Image out(newWidth, newHeight, this->channels);
			float sx = (float)this->width / newWidth;
			float sy = (float)this->height / newHeight;
			for (int y = 0; y < newHeight; ++y) {
				float fy = (std::max)(0.0f, (y + 0.5f) * sy - 0.5f);
				int y0 = (std::min)((int)fy, this->height - 1);
				int y1 = (std::min)(y0 + 1, this->height - 1);
				float wy = fy - y0;
				for (int x = 0; x < newWidth; ++x) {
					float fx = (std::max)(0.0f, (x + 0.5f) * sx - 0.5f);
					int x0 = (std::min)((int)fx, this->width - 1);
					int x1 = (std::min)(x0 + 1, this->width - 1);
					float wx = fx - x0;
					for (int c = 0; c < this->channels; ++c) {
						float top = this->at(x0, y0, c) * (1.0f - wx) + this->at(x1, y0, c) * wx;
						float bottom = this->at(x0, y1, c) * (1.0f - wx) + this->at(x1, y1, c) * wx;
						out.at(x, y, c) = (uint8_t)(top * (1.0f - wy) + bottom * wy + 0.5f);
					}
				}
			}
			return out;
		}
		// 채널 수 변환 (흑백 ↔ 컬러, 알파 제거)
		Image withChannels(int target) const {
			if (target == this->channels)
				return *this;
			Image out(this->width, this->height, target);
			for (int y = 0; y < this->height; ++y) {
				for (int x = 0; x < this->width; ++x) {
					uint8_t rgb[3];
					if (this->channels >= 3) {
						rgb[0] = this->at(x, y, 0);
						rgb[1] = this->at(x, y, 1);
						rgb[2] = this->at(x, y, 2);
					}
					else {
						rgb[0] = this->at(x, y, 0);
						rgb[1] = rgb[0];
						rgb[2] = rgb[0];
					}
					if (target == 1) {
						out.at(x, y, 0) = (uint8_t)(0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2] + 0.5f);
						continue;
					}
					for (int c = 0; c < target; ++c) {
						if (c < 3)
							out.at(x, y, c) = rgb[c];
						else
							out.at(x, y, c) = 255;
					}
				}
			}
			return out;
		}
		// 모델 입력 모양(C, H, W)에 맞춰 0~1 float, CHW 순서로 채운다.
		void toCHW(float* out, int c, int h, int w) const {
			Image fitted = this->withChannels(c).resized(w, h);
			for (int ch = 0; ch < c; ++ch)
				for (int y = 0; y < h; ++y)
					for (int x = 0; x < w; ++x)
						out[((size_t)ch * h + y) * w + x] = fitted.at(x, y, ch) / 255.0f;
		}
		// CHW 0~1 float → 이미지 (MNIST 샘플을 파일로 저장할 때)
		static Image fromCHW(const float* data, int c, int h, int w) {
			Image out(w, h, c);
			for (int ch = 0; ch < c; ++ch) {
				for (int y = 0; y < h; ++y) {
					for (int x = 0; x < w; ++x) {
						float v = data[((size_t)ch * h + y) * w + x] * 255.0f;
						out.at(x, y, ch) = (uint8_t)(std::min)(255.0f, (std::max)(0.0f, v + 0.5f));
					}
				}
			}
			return out;
		}
	private:
		static void writeLE(uint8_t* p, uint32_t v) {
			p[0] = (uint8_t)(v & 0xff);
			p[1] = (uint8_t)((v >> 8) & 0xff);
			p[2] = (uint8_t)((v >> 16) & 0xff);
			p[3] = (uint8_t)((v >> 24) & 0xff);
		}
		static uint32_t readLE(const uint8_t* p) {
			return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
		}
		// stb 를 끈 경우를 위한 최소 BMP(24/32비트 비압축) 디코더
		static Image decodeBMP(const std::vector<uint8_t>& bytes) {
			Image image;
			if (bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M')
				return image;
			uint32_t offset = readLE(&bytes[10]);
			int w = (int)readLE(&bytes[18]);
			int h = (int)readLE(&bytes[22]);
			int bits = bytes[28] | (bytes[29] << 8);
			if (bits != 24 && bits != 32)
				return image;
			bool topDown = h < 0;
			h = std::abs(h);
			int step = bits / 8;
			int rowBytes = (w * step + 3) / 4 * 4;
			if (bytes.size() < offset + (size_t)rowBytes * h)
				return image;
			image = Image(w, h, 3);
			for (int y = 0; y < h; ++y) {
				int srcRow = h - 1 - y;
				if (topDown)
					srcRow = y;
				const uint8_t* src = &bytes[offset + (size_t)srcRow * rowBytes];
				for (int x = 0; x < w; ++x) {
					image.at(x, y, 0) = src[x * step + 2];
					image.at(x, y, 1) = src[x * step + 1];
					image.at(x, y, 2) = src[x * step + 0];
				}
			}
			return image;
		}
	};

	inline bool isImageFile(const std::filesystem::path& path) {
		std::string ext = path.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
		const char* known[] = { ".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif", ".psd", ".pgm", ".ppm", ".pnm" };
		for (const char* k : known) {
			if (ext == k)
				return true;
		}
		return false;
	}
}

#endif
