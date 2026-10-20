# SimpleCNN

**레이어만 쌓으면 바로 학습되는 C++ CNN 라이브러리.**

| 파일 | 내용 |
|---|---|
| `SimpleCNN.h` | 라이브러리 전체. **이것만 include 하면 끝** |
| `SimpleCNN.cuh` | CUDA 커널. SimpleCNN.h 가 알아서 읽어 들이므로 따로 신경 쓸 필요 없음 |

두 파일을 프로젝트 폴더에 복사하고 `#include "SimpleCNN.h"` 한 줄이면 됩니다.
외부 라이브러리(OpenCV, CUDA 링크, 빌드 설정)가 필요 없고, 같은 실행 파일이 GPU 가 있으면 CUDA 로, 없으면 CPU 로 돕니다.

```cpp
#include "SimpleCNN.h"
using namespace cnn;

int main() {
    Dataset train = Dataset::loadFolder("images/train", 64, 64);   // 폴더 이름 = 클래스

    CNN model(Device::CUDA);                       // GPU 사용 (없으면 자동으로 CPU)
    model.add(new Conv(3, 16, 1, 1, ACTIVATION::ReLU));
    model.add(new Pooling(POOLING::Max));
    model.add(new Conv(3, 32, 1, 1, ACTIVATION::ReLU));
    model.add(new Pooling(POOLING::Max));
    model.add(new FullyConnected(128, ACTIVATION::ReLU));
    model.add(new FullyConnected(10, ACTIVATION::Softmax));

    model.compile(OPTIMIZER::Adam, LOSS::CategoricalCrossentropy);
    model.fit(train, 5, 64);                       // epochs, batch size

    int label = model.predict("cat.bmp");          // 이미지 파일로 바로 예측
    model.save("model.scnn");
}
```

입력 이미지 크기, 채널 수, 레이어 사이의 크기는 **데이터를 보고 자동으로** 맞춰집니다.
사용자는 블럭(레이어)을 쌓기만 하면 됩니다.

---

## 1. 바로 실행해 보기 (Visual Studio)

1. `SimpleCNN.sln` 을 엽니다. (VS 2019 / 2022 / 2026 자동 인식)
2. 구성을 `Release | x64` 로 바꾸고 **F5**.
3. `MnistTest` 가 MNIST 손글씨 데이터를 내려받아 **CUDA 로 학습 → 평가 → 저장 → 이미지 파일 예측** 까지 진행합니다.

| 프로젝트 | 하는 일 |
|---|---|
| `MnistTest` (시작 프로젝트) | 실제 데이터로 전체 흐름 테스트. 디버깅 인자: Debug=`cuda --epochs 1 --train 10000`, Release=`cuda` |
| `UnitTest` | 역전파 수치 검증, im2col/FFT 결과 비교, 저장/불러오기, PNG/BMP/RAW 읽기와 폴더 학습, CPU↔CUDA 결과 비교 |

평소처럼 중단점을 걸고 디버깅하면 됩니다. 인자는 *프로젝트 속성 → 디버깅 → 명령 인수* 에서 바꿉니다.

```
MnistTest cuda                         GPU 로 학습
MnistTest cpu --epochs 1 --train 10000 CPU 로 1만 장만
MnistTest cuda --image my_digit.png    학습 후 내가 그린 숫자 예측
MnistTest cuda --algo fft              합성곱 방식 고정 (auto / im2col / fft)
MnistTest cuda --folder photos         MNIST 대신 내 이미지 폴더로 학습 (photos/클래스이름/*.png, *.bmp, *.raw)
MnistTest cuda --folder photos --size 64 --channels 3 --epochs 20 --image test.png
```

`--folder` 는 이미지를 `--size` 크기(기본 32)로 맞춰 읽고, 80% 로 학습해 나머지 20% 로 평가한 뒤 `data/folder_model.scnn` 에 저장합니다.

Linux / macOS 는 CMake 를 씁니다.
```
cmake -B build && cmake --build build -j
./build/MnistTest cuda
```

## 2. 블럭(레이어) 종류

| 블럭 | 만드는 법 | 설명 |
|---|---|---|
| Conv | `Conv(필터크기, 필터개수, stride, padding, 활성화)` | 합성곱. `Conv(3, 16, 1, 1, ReLU)` = 3x3 필터 16개 |
| Pooling | `Pooling(POOLING::Max / Average / Min, 크기=2, stride=크기)` | 크기 줄이기 |
| Padding | `Padding(크기)` | 테두리에 0 채우기 |
| Activation | `Activation(ACTIVATION::ReLU)` | 활성화만 따로 쓰고 싶을 때 |
| Flatten | `Flatten()` | 펼치기. FullyConnected 가 알아서 펼치므로 생략 가능 |
| FullyConnected | `FullyConnected(유닛 수, 활성화)` | 완전연결층. 마지막은 보통 `(클래스 수, Softmax)` |

- **활성화**: `None, ReLU, Leaky_ReLU, ELU, Sigmoid, TanH, Softmax, Maxout`
- **옵티마이저**: `GD, Mini_SGD, Momentum, Adagrad, RMSProp, Adam, Nadam` (학습률은 자동, `setLearningRate` 로 변경)
- **손실**: `CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy, MSE, RMSE`

`model.add(Conv(3, 16))` 처럼 `new` 없이 값으로 넘겨도 됩니다. `new` 로 넘긴 레이어는 CNN 이 알아서 지웁니다.

## 3. 데이터 넣기

```cpp
Dataset a = Dataset::loadFolder("images", 64, 64, 3);              // images/고양이/*.bmp, images/강아지/*.bmp
Dataset b = Dataset::loadFolder("huge", 224, 224, 3, true);        // streaming: 파일 경로만 들고 학습 중에 읽음
Dataset c = Dataset::loadFiles("imgs", "labels.csv", 64, 64);      // 이름순 이미지 + 라벨 CSV
Dataset d = Dataset::loadMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
Dataset e = Dataset::fromVectors(v4d, labels);                     // 예전 V4D 방식 그대로
auto [train, test] = a.split(0.2f);                                // 80% 학습 / 20% 테스트
```

아래 이미지 형식은 외부 라이브러리 없이 직접 읽습니다. 크기와 채널이 제각각이어도 `loadFolder` 에 준 크기로 맞춥니다.

| 형식 | 지원 범위 |
|---|---|
| PNG | 흑백 / RGB / 팔레트 / 알파, 1·2·4·8·16 비트, 인터레이스 (inflate 직접 구현) |
| BMP | 1·4·8 비트 팔레트 (RLE 압축 포함), 16·24·32 비트 (비트필드 포함), OS/2 헤더 |
| PGM / PPM | P5 (흑백), P6 (컬러) |
| RAW | 헤더 없는 8비트 HWC (RGB 순서). 파일 이름에 크기: `cat_64x48x3.raw`. 이름에 크기가 없으면 정사각형으로 추정 |

저장은 `image.savePNG()`, `image.saveBMP()`, `image.saveRaw()` 로 합니다.
**JPG / GIF / TGA** 가 필요하면 [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h) (퍼블릭 도메인 파일 하나) 를 `SimpleCNN.h` 옆에 두기만 하면 자동으로 인식합니다.

예전 README 방식도 그대로 동작합니다.
```cpp
CNN* cnn = new CNN(images, labels);   // V4D, V1D
cnn->add(new Conv(3, 3, 1, 1, ACTIVATION::TanH));
...
cnn->compile(OPTIMIZER::Mini_SGD, LOSS::CategoricalCrossentropy);
cnn->fit(epochs, batchSize);
cnn->accuracy();
```

## 4. 내부 구조

`SimpleCNN.h` 는 위에서 아래로 읽으면 전체가 이해되도록 8개 구역으로 되어 있습니다.

| 구역 | 내용 |
|---|---|
| 1. 기본 정의 | enum, Shape(텐서 모양), parallelFor(멀티코어 반복) |
| 2. GPU 런타임 | CUDA 드라이버 / NVRTC / cuBLAS 를 실행 중에 연결, GPU 메모리 풀 |
| 3. Tensor | 연속 메모리 한 덩어리 (CPU 또는 GPU) |
| 4. 연산 | GEMM, im2col, FFT 합성곱, 풀링, 활성화 (`cpu::` / `gpu::`) |
| 5. 레이어 | Conv, Pooling, Padding, Activation, Flatten, FullyConnected |
| 6. 옵티마이저 | SGD ~ Nadam |
| 7. 이미지 / 데이터셋 | inflate, PNG/BMP/PGM/PPM/RAW 읽기·저장, MNIST, 폴더 데이터셋 |
| 8. CNN 모델 | add → compile → fit → predict / save / load |

모든 블럭(레이어)은 같은 인터페이스 3개만 가집니다.

| 함수 | 역할 |
|---|---|
| `build(입력모양)` | 앞 블럭 출력 모양을 받아 자기 출력 모양을 정하고 가중치를 만든다 |
| `forward(x, y)` | 순전파 |
| `backward(x, y, dy, dx)` | 역전파 (dx 와 자기 가중치 기울기) |

새 레이어를 만들 때도 `Layer` 를 상속해서 이 세 개만 구현하면 CNN 에 그대로 끼워집니다.

### CUDA 가 동작하는 방식

`Device::CUDA` 를 주면 실행 중에 아래 순서로 GPU 를 준비합니다. 실패하면 이유를 출력하고 CPU 로 계속합니다.

1. **드라이버 찾기**: `nvcuda.dll` (그래픽 드라이버에 포함)
2. **NVRTC 찾기**: `CUDA_PATH` 또는 기본 설치 폴더에서 `nvrtc64_*.dll` (CUDA Toolkit 에 포함)
3. **커널 컴파일**: 지금 꽂힌 GPU 의 아키텍처(예: RTX 4070 = sm_89)로 커널을 즉석 컴파일
4. **cuBLAS 연결**: 있으면 행렬곱에 사용, 없으면 내장 타일링 GEMM 커널 사용

그래서 프로젝트에 CUDA 빌드 설정이나 `.lib` 링크가 전혀 없고, GPU 가 없는 PC 에서도 같은 exe 가 실행됩니다.
환경변수 `SIMPLECNN_DEVICE=1` 은 두 번째 GPU 선택, `SIMPLECNN_NO_CUBLAS=1` 은 cuBLAS 끄기입니다.

### 합성곱 계산 방식 (`CONV_ALGO`)

| 방식 | 원리 | 유리한 경우 |
|---|---|---|
| `Im2col` | 이미지 조각을 행렬로 펼쳐 **행렬곱(GEMM) 한 번**으로 계산 | 대부분 (3x3 등 작은 필터) |
| `FFT` | 주파수 영역으로 바꾸면 합성곱이 **곱셈**이 됨 | 큰 필터(5x5, 7x7 이상), 큰 이미지 |
| `Direct` | 정의 그대로 반복문 | 검증용 |
| `Auto` (기본) | 첫 배치에서 im2col 과 FFT 를 **실제로 재보고** 빠른 쪽을 계속 사용 | |

`model.setConvAlgorithm(CONV_ALGO::FFT)` 또는 `new Conv(..., CONV_ALGO::FFT)` 로 고정할 수 있습니다.
GPU 는 im2col + cuBLAS 를 사용합니다. 역전파는 모든 경우 im2col + GEMM 입니다.

### 메모리 최적화

| 기법 | 효과 |
|---|---|
| 중첩 `vector` 대신 연속 메모리 `Tensor` | 행마다 따로 할당하던 오버헤드 제거, 캐시 효율 |
| 버퍼 재사용 (커질 때만 재할당) | 배치마다 할당/해제가 없음 |
| 기울기 버퍼 2개만 번갈아 사용 | 레이어가 많아져도 기울기 메모리는 일정 |
| im2col 작업 버퍼를 모든 레이어가 공유 + 상한(`setWorkspaceLimit`) | 큰 이미지는 배치를 쪼개서 처리 (GPU 메모리 부족 방지) |
| 이미지 데이터는 uint8 로 보관 | float 대비 1/4 |
| `streaming` 데이터셋 | 수십 GB 데이터도 RAM 거의 사용 안 함 |
| 활성화 미분을 출력값으로 계산 | 활성화 전 값을 따로 저장하지 않음 |
| Max/Min 풀링은 선택 위치(인덱스)만 저장 | 예전 0/1 마스크 V4D 대비 1/4 |
| 옵티마이저 상태는 필요한 것만 | SGD 0, Momentum 1벌, Adam 2벌 |
| GPU 메모리 풀 | cudaMalloc 반복 호출 없음 |

`model.summary()` 로 레이어별 출력 크기, 파라미터 수, 예상 메모리를 볼 수 있습니다.

## 5. 검증 결과

- `UnitTest`: GEMM 4가지 전치 조합, im2col = FFT = Direct, 모든 레이어/활성화 역전파 수치 미분 일치, 저장/불러오기 일치 (35개 검사 통과, AddressSanitizer/UBSan 오류 0)
- CUDA 커널: NVRTC 로 sm_75 / sm_86 / sm_120 / compute_52 컴파일 확인, 커널 에뮬레이션으로 7가지 구성에서 CPU 와 손실값 일치 확인
- `MnistTest` (CPU 4코어, 1 epoch, 6만 장): 테스트 정확도 **98.13%**, 메모리 최대 14.5MB

## 요구 사항

- C++17 컴파일러 (Visual Studio 2019 이상, GCC 9+, Clang 10+)
- GPU 사용 시: NVIDIA 그래픽 드라이버 + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (버전 무관, 11.2 이상 권장)
- OpenCV 불필요

## License

MIT License
