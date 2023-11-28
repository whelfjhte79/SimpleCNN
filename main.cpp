#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<iostream>
#include"WinVisualization.h" // �ð�ȭ ��Ʈ
//#define _CUDA_GPU_ // GPU ���
#include"SimpleCNN.h"



#ifdef _CUDA_GPU_
#include"kernel.cuh"
#endif //__CUDACC__

using namespace cv;
using namespace std;
using namespace img_read;
using namespace cnn;





//#include"test.h"
//#define __CUDACC__
// ���κκ�
/*
#if !defined(TEST) && !defined(TEST2)

int sum_int(int a, int b);
int sum_int(int a, int b) {
	int c;
	c = a + b;
	return c;
}



#ifdef __CUDACC__
int main() {
	cout << "�׽�Ʈ";
	return 0;
}

#else
int main(void) {
	int a = 7, b = 5, c, d;
	
	c = sum_int(7, 5);
	GPU_TEST gpu_test;
	gpu_test.sum_cuda(a, b, &d);
	
	printf_s("CPU : %d + %d = %d  \n",a,b,c);
	printf_s("GPU : %d + %d = %d  \n",a,b,d);


	return 0;
}
#endif
*/

#ifdef _CUDA_GPU_
int main(void) {
	std::cout << "�׽�Ʈ";
	V4D dataset = V4D(1000, V3D(3, V2D(100, V1D(100, 0.0f))));

	RandomGen* randFloat = new RandomGen(0.0f, 1.0f);
	RandomGen* randInt = new RandomGen(0, 9);
	V1D label = V1D(1000, 0.0f);
	for (int i = 0; i < dataset.size(); ++i) {
		randFloat->randGen(dataset[i], false);
		//randInt->randGenInt(label[i]);
	}
	CNN* cnn = new CNN(dataset, label);

	cnn->add(new Conv(3, 8, 1, 1, ACTIVATION::TanH));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(2));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Min));
	//cnn->add(new Pooling(POOLING::Min));
	//cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Flatten());
	cnn->add(new FullyConnected(64, ACTIVATION::Maxout));
	cnn->add(new FullyConnected(32, ACTIVATION::Sigmoid));
	cnn->add(new FullyConnected(16, ACTIVATION::ReLU));

	cnn->add(new FullyConnected(10, ACTIVATION::Softmax));


	cnn->compile(OPTIMIZER::Momentum, LOSS::CategoricalCrossentropy);
	cnn->fit(1, 100);
	


	//ConvGPU gpu;
	
	//PrintTest p;
	//p.printTestV4D(gpu.calculateGPU(dataset), "CONV GPU");
	//gpu.filter2col()

	
	//WinApp* app = new WinApp({ 50 , 100 , 150 , 200 , 250, 300 }, { 100 , 50 , 200 , 150 , 250, 400 }, "X Axis", "Y Axis");
	//app->Run();

	return 0;
}


#else
int main(void) {
	//Directory* dirent = new Directory("C:\\Users\\�̻��\\source\\repos\\SimpleCNN_image\\SimpleCNN_image\\�̹����׽�Ʈ\\Humans", img_read::FILENAME_EXTENSION::JPG);

	V4D dataset = V4D(50, V3D(3, V2D(100, V1D(100, 0.0f))));
	
	RandomGen* randFloat = new RandomGen(0.0f, 1.0f);
	RandomGen* randInt = new RandomGen(0,9);
	V1D label = V1D(50, 0.0f);
	for (int i = 0; i < dataset.size(); ++i) {
		randFloat->randGen(dataset[i], false);
		//randInt->randGenInt(label[i]);
	}
	CNN* cnn = new CNN(dataset, label);
	//cnn->splitTrainTest(0.3);
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::TanH));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(2));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Flatten());
	cnn->add(new FullyConnected(64, ACTIVATION::Maxout));
	cnn->add(new FullyConnected(32, ACTIVATION::Sigmoid));
	cnn->add(new FullyConnected(16, ACTIVATION::ReLU));

	cnn->add(new FullyConnected(10, ACTIVATION::Softmax));

	cnn->compile(OPTIMIZER::Momentum, LOSS::CategoricalCrossentropy);
	cnn->fit(1, 5);

	
	cnn->save("cnn_test.txt");
	cnn->load("cnn_test.txt");
	//V1D predictions = cnn->predict(dirent->getImageSet());
	
	//cnn->accuracy(predictions, label->getLabel());




	return 0;
}

/*
int main(void) {

	Directory* dirent = new Directory("C:\\Users\\�̻��\\source\\repos\\SimpleCNN_image\\SimpleCNN_image\\�̹����׽�Ʈ\\Humans",img_read::FILENAME_EXTENSION::JPG);
	FileRead* label = new FileRead("C:\\Users\\�̻��\\source\\repos\\SimpleCNN_image\\SimpleCNN_image\\HumanLabel.csv");

	V4D dataset = dirent->getImageSet();
	
	std::cout << "�α� Conv �Էµ�����:" << dataset.size() << " ";
	std::cout << dataset[0].size() << " ";
	std::cout << dataset[0][0].size() << " ";
	std::cout << dataset[0][0][0].size() << "\n";
	
	CNN* cnn = new CNN(dirent->getImageSet(), label->getLabel());
	//cnn->splitTrainTest(0.3);
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::TanH)); 
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::ReLU));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Padding(1));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Average));
	cnn->add(new Pooling(POOLING::Max));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Pooling(POOLING::Min));
	cnn->add(new Flatten());
	cnn->add(new FullyConnected(256, ACTIVATION::Maxout));
	cnn->add(new FullyConnected(128, ACTIVATION::Sigmoid));
	cnn->add(new FullyConnected(64, ACTIVATION::ReLU));

	cnn->add(new FullyConnected(10, ACTIVATION::Softmax));

	cnn->compile(OPTIMIZER::Momentum, LOSS::CategoricalCrossentropy);
	cnn->fit(1,50);

	cnn->save("cnn_test.txt");
	cnn->load("cnn_test.txt");
	V1D predictions = cnn->predict(dirent->getImageSet());

	cnn->accuracy(predictions, label->getLabel());
	
	

	return 0;
}*/
#endif
#ifdef TEST
#include"test.h"
/*
* ����: �� ���̾ calculate �Լ� �˻�
* �˻�Ǹ� ��ġ��
* ���� ��ȯ �� �ȵŴ��� Ȯ��

*/

#include <random>
#include <vector>
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(1, 10000);

int main(void) {
	V2D testVec2D = { {0.3f,-1.4f,2.0f,-1.7f}, 
					  {-2.3f,1.1f,1.9f,0.5f} 
					};
	V2D label2D = { {0.0f,0.0f,1.0f,0.0f},
					{1.0f,0.0f,2.0f,4.0f}
	};
	
	CNN* cnn = new CNN();

	cout<< cnn->derivativeLossFunc(&testVec2D, &label2D);



	return 0;
}

#endif
#ifdef TEST2
#include"test.h"


int main(void) {
	test();
	return 0;
}


#endif