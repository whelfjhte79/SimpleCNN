// 예제용 도우미: 파일 내려받기(curl) + gzip 풀기
// gzip 은 zlib 없이 풀 수 있도록 inflate(RFC 1951)를 직접 구현했다.
// Windows 10 이상은 curl.exe 가 기본으로 들어 있다.
#ifndef SIMPLE_CNN_DOWNLOAD_H
#define SIMPLE_CNN_DOWNLOAD_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace download {

	class Inflater {
	private:
		struct Huffman {
			std::vector<short> count;
			std::vector<short> symbol;
		};
		const uint8_t* in;
		size_t inSize;
		size_t pos;
		uint32_t bitBuffer = 0;
		int bitCount = 0;
		std::vector<uint8_t>& out;

		int bits(int need) {
			uint32_t value = this->bitBuffer;
			while (this->bitCount < need) {
				if (this->pos >= this->inSize)
					throw std::runtime_error("gzip 데이터가 중간에 끊겼습니다.");
				value |= (uint32_t)this->in[this->pos++] << this->bitCount;
				this->bitCount += 8;
			}
			this->bitBuffer = value >> need;
			this->bitCount -= need;
			return (int)(value & ((1u << need) - 1));
		}
		static void build(Huffman& h, const short* lengths, int n) {
			h.count.assign(16, 0);
			h.symbol.assign(n, 0);
			for (int s = 0; s < n; ++s)
				h.count[lengths[s]]++;
			h.count[0] = 0;
			short offsets[16] = { 0 };
			for (int len = 1; len < 15; ++len)
				offsets[len + 1] = offsets[len] + h.count[len];
			for (int s = 0; s < n; ++s) {
				if (lengths[s] != 0)
					h.symbol[offsets[lengths[s]]++] = (short)s;
			}
		}
		int decode(const Huffman& h) {
			int code = 0;
			int first = 0;
			int index = 0;
			for (int len = 1; len <= 15; ++len) {
				code |= this->bits(1);
				int count = h.count[len];
				if (code - count < first)
					return h.symbol[index + (code - first)];
				index += count;
				first += count;
				first <<= 1;
				code <<= 1;
			}
			throw std::runtime_error("잘못된 허프만 코드");
		}
		void stored() {
			this->bitBuffer = 0;
			this->bitCount = 0;
			if (this->pos + 4 > this->inSize)
				throw std::runtime_error("gzip 데이터가 중간에 끊겼습니다.");
			int length = this->in[this->pos] | (this->in[this->pos + 1] << 8);
			this->pos += 4;
			if (this->pos + length > this->inSize)
				throw std::runtime_error("gzip 데이터가 중간에 끊겼습니다.");
			this->out.insert(this->out.end(), this->in + this->pos, this->in + this->pos + length);
			this->pos += length;
		}
		void codes(const Huffman& lengthCode, const Huffman& distCode) {
			static const short lengthBase[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
			static const short lengthExtra[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
			static const short distBase[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
			static const short distExtra[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
			while (true) {
				int symbol = this->decode(lengthCode);
				if (symbol < 256) {
					this->out.push_back((uint8_t)symbol);
					continue;
				}
				if (symbol == 256)
					return;
				symbol -= 257;
				if (symbol >= 29)
					throw std::runtime_error("잘못된 길이 코드");
				int length = lengthBase[symbol] + this->bits(lengthExtra[symbol]);
				int distSymbol = this->decode(distCode);
				if (distSymbol >= 30)
					throw std::runtime_error("잘못된 거리 코드");
				size_t dist = (size_t)distBase[distSymbol] + this->bits(distExtra[distSymbol]);
				if (dist > this->out.size())
					throw std::runtime_error("잘못된 거리");
				size_t from = this->out.size() - dist;
				for (int i = 0; i < length; ++i)
					this->out.push_back(this->out[from + i]);
			}
		}
		void fixed() {
			short lengths[320];
			int s = 0;
			for (; s < 144; ++s)
				lengths[s] = 8;
			for (; s < 256; ++s)
				lengths[s] = 9;
			for (; s < 280; ++s)
				lengths[s] = 7;
			for (; s < 288; ++s)
				lengths[s] = 8;
			Huffman lengthCode;
			build(lengthCode, lengths, 288);
			for (s = 0; s < 30; ++s)
				lengths[s] = 5;
			Huffman distCode;
			build(distCode, lengths, 30);
			this->codes(lengthCode, distCode);
		}
		void dynamic() {
			static const short order[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
			int nlen = this->bits(5) + 257;
			int ndist = this->bits(5) + 1;
			int ncode = this->bits(4) + 4;
			short lengths[320] = { 0 };
			for (int i = 0; i < ncode; ++i)
				lengths[order[i]] = (short)this->bits(3);
			Huffman lencode;
			build(lencode, lengths, 19);
			int index = 0;
			while (index < nlen + ndist) {
				int symbol = this->decode(lencode);
				if (symbol < 16) {
					lengths[index++] = (short)symbol;
					continue;
				}
				short value = 0;
				int repeat = 0;
				if (symbol == 16) {
					if (index == 0)
						throw std::runtime_error("잘못된 반복 코드");
					value = lengths[index - 1];
					repeat = 3 + this->bits(2);
				}
				else if (symbol == 17) {
					repeat = 3 + this->bits(3);
				}
				else {
					repeat = 11 + this->bits(7);
				}
				if (index + repeat > nlen + ndist)
					throw std::runtime_error("잘못된 길이 표");
				while (repeat-- > 0)
					lengths[index++] = value;
			}
			Huffman lengthCode;
			build(lengthCode, lengths, nlen);
			Huffman distCode;
			build(distCode, lengths + nlen, ndist);
			this->codes(lengthCode, distCode);
		}
	public:
		Inflater(const uint8_t* data, size_t size, size_t start, std::vector<uint8_t>& output) : in(data), inSize(size), pos(start), out(output) {}

		void run() {
			int last = 0;
			while (last == 0) {
				last = this->bits(1);
				int type = this->bits(2);
				if (type == 0)
					this->stored();
				else if (type == 1)
					this->fixed();
				else if (type == 2)
					this->dynamic();
				else
					throw std::runtime_error("잘못된 블록 형식");
			}
		}
	};

	// .gz 한 파일을 풀어 outPath 로 저장
	inline void gunzip(const std::filesystem::path& gzPath, const std::filesystem::path& outPath) {
		std::ifstream file(gzPath, std::ios::binary);
		std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		if (data.size() < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 8)
			throw std::runtime_error("gzip 파일이 아닙니다: " + gzPath.string());
		int flags = data[3];
		size_t pos = 10;
		if (flags & 4)
			pos += 2 + (data[pos] | (data[pos + 1] << 8));
		if (flags & 8) {
			while (data[pos] != 0)
				++pos;
			++pos;
		}
		if (flags & 16) {
			while (data[pos] != 0)
				++pos;
			++pos;
		}
		if (flags & 2)
			pos += 2;
		std::vector<uint8_t> output;
		output.reserve(data.size() * 4);
		Inflater inflater(data.data(), data.size(), pos, output);
		inflater.run();
		std::ofstream result(outPath, std::ios::binary);
		result.write(reinterpret_cast<const char*>(output.data()), (std::streamsize)output.size());
	}

	inline bool downloadFile(const std::string& url, const std::filesystem::path& path) {
		std::string command = "curl -L --fail --silent --show-error -o \"" + path.string() + "\" \"" + url + "\"";
		int result = std::system(command.c_str());
		std::error_code ec;
		return result == 0 && std::filesystem::exists(path, ec) && std::filesystem::file_size(path, ec) > 0;
	}

	// MNIST 4개 파일을 dir 에 준비한다. 이미 있으면 건너뛴다.
	inline bool prepareMnist(const std::filesystem::path& dir) {
		const char* names[] = { "train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte" };
		const char* mirrors[] = { "https://storage.googleapis.com/cvdf-datasets/mnist/", "https://ossci-datasets.s3.amazonaws.com/mnist/" };
		std::error_code ec;
		std::filesystem::create_directories(dir, ec);
		for (const char* name : names) {
			std::filesystem::path target = dir / name;
			if (std::filesystem::exists(target, ec))
				continue;
			std::filesystem::path gz = dir / (std::string(name) + ".gz");
			bool ok = false;
			for (const char* mirror : mirrors) {
				std::printf("내려받는 중: %s%s.gz\n", mirror, name);
				if (downloadFile(std::string(mirror) + name + ".gz", gz)) {
					ok = true;
					break;
				}
			}
			if (!ok)
				return false;
			gunzip(gz, target);
			std::filesystem::remove(gz, ec);
		}
		return true;
	}
}

#endif
