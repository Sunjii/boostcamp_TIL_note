# Computer Vision - Image Classification

## Overview

AI에는 사람의 지능을 컴퓨터 시스템으로 구현한 것이다. 여기서 지능은 인지능력과 지각능력을 포함한 사고능력을 의미한다. 즉, 시각역시 인공지능을 구현하기 위해 필요한 영역인 것이다. 사람으로 예를 들어 표현하자면, 아기시절 세상을 배우기 위해서는 시각과 청각, 미각 등 오감을 전부 이용하여 배우게 된다. 이를 통해 세상과 상호작용을 하고 인과관계를 배우며 사고를 익히기 시작한다. 즉, 지각 능력이 지능에 있어서 가장 중요한 시작인 것이다.

![](001.png)
![](002.png)
![](003.png)

따라서 Computer Vision은 기계에게 이미지를 어떻게 볼지를 가르치는 분야이다.

- Visual perception & intelligence
  - input: visual data (image or video)
- visual perception의 종류
  - Color
  - Motion
  - 3D
  - Semantic-level
  - Social perception (emotion)
  - Visuomotor
  - etc
- 또, 사람이 보는 방식도 포함한다!
  - 사람의 시각능력은 불완전하다.
  - 예를 들어 착시나 illusion image등이 있다. 이를 기계도 사람처럼 볼 수 있도록 학습하는 것도 포함된다.
  - 혹은 적당히 불완전한 이미지도 뇌를 거쳐 보정하여 보는 경우도 있다.
  - 사람의 구조(장단점이 있다)를 모방하는 것이 시작이기 때문

### Machine Learning vs Deep Learning

사람이 직접 특징을 추출하는 것 vs gradient desecent를 통한 특징 추출

![](004.png)
![](005.png)

deep learning의 부활이 일어나면서 40년간 정체되었던 computer vision이 크게 발전하기 시작했다. 이제는 실제 서비스가 가능한 단계까지 발전했으며 최근에는 computer vision 학술지가 Top 5에 들어가기도 했다.

## Image Classification

분류기 (Classifier)

![](006.png)

- 어떻게 구현할까?
  - 만약 세상의 모든 지식(정보)를 가지고 있다면?
  - 그저 갖고 있는 정보에서 찾는, 검색 문제가 된다.
    - k Nearest Neighbors Problem (k-NN)
    - ![](007.png)
  - 다만 세상의 모든 정보를 갖고 있다는 것은 허황된 이야기다...
  - 또, 영상(이미지)간의 유사도는 어떻게 계산할 것인가도 중요한 문제다. 이를 계산하는 것은 시스템 복잡도가 상당하므로 어려운 문제다.
  - 이 모든 것을 Neural Network에 녹여 학습을 해보면 어떨까?

### Convolutional Neural Network (CNN)

- Single layer neural network (fully-connected) : FC layer

![](008.png)

모든 픽셀을 서로 다른 가중치로 weighted sum (내적)을 하고, Activation fuction을 통해 분류 score를 출력하는 가장 간단한 모델이다. 간단한만큼 한계도 명확하게 존재한다.

![](009.png)

Score를 살펴보자. 하단의 그림은 해당하는 class들의 평균적인 score를 이미지화 한 것이다. 여기서 두가지 문제가 있다. 우선 첫번째는 layer가 한층이라 매우 단순하므로 평균이미지외를 표현하기가 너무 힘들다는 단점이 있다.

![](010.png)

두번째 문제로는 학습시 사용한 이미지와 다른 이미지가 들어가는 경우 발생한다. 같은 object이지만 약간의 다른 변화만 있더라도 다른 결과를 내놓게 된다.

이러한 Fully-connected layer를 극복하기 위해 등장한 것이 바로 CNN이다.

![](011.png)

- CNN
  - locally connected neural network
  - 하나의 특징을 뽑기 위해 모든 픽셀을 고려하는 fc 레이어와는 다르다
  - 하나의 특징을 위해서 국부적인 픽셀만을 고려한다.
    - 국부적으로 parameter의 숫자가 획기적으로 줄게된다.
    - 국부적으로 본 영역들 간의 연결성도 놓치지 않는다. (슬라이딩 윈도우)
    - Over-fitting도 방지된다.
  - 이런 특징 덕분에 CNN은 다양한 CV task에서 backbone으로 사용된다.

![](012.png)

## Brief History

![](013.png)

### AlexNet

CNN 구조가 처음 고안된 것은 1998년. 당시에는 CNN 2개정도의 간단한 구조 (LeNet-5)

- Conv - Pool - Conv - Pool - FC - FC
- 5x5 filter with stride 1

AlexNet은 이 구조에서 많은 모티브를 받았다. 차이점은 다음과 같다.

![](014.png)

- 7개의 히든 레이어 (60m의 파라미터)
- ImageNet을 이용한 학습 (1.2milion 개의 이미지)
- 더 나은 activation function인 ReLU와 dropout 테크닉 적용
- 모델 구조가 2부분으로 나뉜 이유는 당시 GPU 성능의 한계가 있었기 때문에 병렬처리를 위함이다.

![](015.png)

- 주의점. MaxPool2d 의 output은 바로 Linear로 들어가지 못 한다. 공간 정보와 채널 정보도 존재하기 때문. 즉, 3D 구조에서 2D 구조(혹은 1D)로 바꾸어야한다. 따라서 3D인 텐서형태를 벡터 형태로 바꿔야한다.
  - 논문에서는 두번째 방법인 flatten을 사용했다.
  - 2048이였는데 왜 4096?
    - 그 당시 학습할때 성능의 문제로 절반씩 하였기 때문에 2048로 적혀있던 것

#### LRN

Local Response Normalization. 국지적인 노말라이제이션.

현재는 사용하지 않는 기법이기 때문에 설명에서 제외했음. 현재는 Batch Normalization을 사용한다.

![](016.png)

그외 AlexNet에서만 썻던 낡은 구조는 11 x 11 conv filter이다. 현재는 이렇게 큰 사이즈의 필터는 사용하지 않는다. Receptiv field란 한 element가 출력되었을 때, 그 element의 출력에 쓰인 field의 영역을 의미한다. (디펜던시가 있다고 봄) 여러 레이어를 거치면서 하나의 결과에 영향을 미친 영역은 점점 커지게 된다.

## VGGNet

지금도 가장 많이 쓰고있는 모델이다.

- 더 깊은 사이즈. 16 ~ 19 레이어
- 더 간단한 구조
  - LRN 사용 안 함
  - 3x3 conv filter, 2x2 max pooling 사용
- 더 좋은 퍼포먼스
- 일반화가 잘 됨 (특징 추출을 잘 한다)

### 상세한 특징

- input
  - 224 x 224 RGB image (AlexNet과 같음)
  - training 이미지의 평균 RGB값을 빼주면서 입력받음 (일종의 normalization)
- Key design choices
  - 3x3 conv filter
  - 2x2 max pooling
  - why?
    - 작은 필터들도 stack을 많이 쌓으면 큰 receptive field 사이즈를 얻을 수 있음.
    - 이는 곧 이미지에서 많은 부분을 고려해서 결론을 내렸다는 뜻이 된다.
    - 더 적은 파라미터를 얻으면서도 더 많은 곳을 고려한 학습이 가능
- 3 FC layer
- ReLU
