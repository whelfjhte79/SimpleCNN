#ifndef IMAGE_READ_H
#define IMAGE_READ_H

#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<dirent.h>
#include<locale.h>
#include<cstring>
#include"SimpleCNN.h"

using namespace cv;
using namespace std;


namespace img_read {
	class Image {
		vector<float> rV, gV, bV;
		vector<vector<float>> rDV, gDV, bDV;
		vector<vector<vector<float>>> channel;
		Mat img;
	public:
		Image() {}
		void imageRead(string fileName) { // PNG
			img = imread(fileName, 1);

			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					float b = (float)img.at<Vec3b>(i, j)[0] / 255.f;
					float g = (float)img.at<Vec3b>(i, j)[1] / 255.f;
					float r = (float)img.at<Vec3b>(i, j)[2] / 255.f;
					rV.push_back(r); gV.push_back(g); bV.push_back(b);

				}
				rDV.push_back(rV); gDV.push_back(gV); bDV.push_back(bV);
				rV.clear();	gV.clear(); bV.clear();
			}
			channel.push_back(rDV);
			channel.push_back(gDV);
			channel.push_back(bDV);
		}
		vector<vector<vector<float>>>& getReadImage() {
			return channel;
		}
		Mat getImageShow() {
			return img;
		}
		void clear() {
			rDV.clear();
			gDV.clear();
			bDV.clear();
			channel.clear();
		}
	};
	enum READ {
		FAILED = false, SUCESS = true
	};
	enum class FILENAME_EXTENSION {
		JPG, JPEG, PNG, BMP
	};
	class Directory {
	private:
		DIR* dir = NULL;
		struct dirent* dirent = NULL;
		const char* filePath;
		const char* fileNameExtension = "";
		std::vector<std::string> fileNameCollection;
		int fileLength = 0;
		img_read::Image imageData;
		vector<vector<vector<vector<float>>>> imageSet;

		void setExtension(FILENAME_EXTENSION fileExtension) {
			switch (fileExtension) {
			case FILENAME_EXTENSION::JPG:
				this->fileNameExtension = ".jpg";
				break;
			case FILENAME_EXTENSION::JPEG:
				this->fileNameExtension = ".jpeg";
				break;
			case FILENAME_EXTENSION::PNG:
				this->fileNameExtension = ".png";
				break;
			case FILENAME_EXTENSION::BMP:
				this->fileNameExtension = ".bmp";
				break;
			}
		}
		void clearfileNameCollection() {
			this->fileNameCollection.clear();
		}

	public:
		Directory(const char* filePath) {
			setlocale(LC_ALL, "");
			this->filePath = filePath;
			setExtension(FILENAME_EXTENSION::JPG);
			setImageSet();
		}
		Directory(const char* filePath, FILENAME_EXTENSION fileExtension) {
			setlocale(LC_ALL, "");
			this->filePath = filePath;
			setExtension(fileExtension);
			setImageSet();
		}
		bool read() {
			this->dir = opendir(filePath);
			if (this->dir) {
				while ((this->dirent = readdir(this->dir))) {
					if (this->dirent->d_type == DT_REG && strstr(this->dirent->d_name, this->fileNameExtension)) {
						printf("%s\n", this->dirent->d_name);
						this->fileNameCollection.push_back(this->dirent->d_name);
						this->fileLength++;
					}
				}
				closedir(dir);
				printf("Found files : %d°³", fileLength);
				return READ::SUCESS;
			}
			else {
				printf("Cannot open directory");
				return READ::FAILED;
			}
		}
		std::vector<std::string> getFileNameCollection() {
			return this->fileNameCollection;
		}
		vector<vector<vector<vector<float>>>>& getImageSet() {
			return this->imageSet;
		}
		void setImageSet() {
			if (this->read()) {
				fileNameCollection = this->getFileNameCollection();
				for (auto iter = fileNameCollection.begin(); iter != fileNameCollection.end(); iter++) {
					imageData.imageRead(*iter);
					imageSet.push_back(imageData.getReadImage());
					imageData.clear();
				}
			}
			else {

			}
		}
	};
	class FileRead { // label read. only int.
	private:
		std::vector<std::string> label;

		std::vector<float> convertStringToFloat(std::vector<std::string> label) {
			std::vector<float> labelFloat;

			for (int i = 0; i < label.size(); i++) {
				labelFloat.push_back(stof(label[i]));
			}
			return labelFloat;
		}
	public:
		FileRead(std::string fileName) {
			this->read(fileName);
		}
		void read(std::string fileName) {
			string str_buf;
			fstream fs;

			fs.open(fileName, ios::in);

			while (!fs.eof()) {
				getline(fs, str_buf, '\n');
				this->label.push_back(str_buf);
			}
			this->label.pop_back();

			fs.close();
		}
		std::vector<float> getLabel() {
			return convertStringToFloat(this->label);
		}
	};
}




#endif