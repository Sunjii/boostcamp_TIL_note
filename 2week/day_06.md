# Intro to PyTorch

파이토치는 딥러닝을 만드는데 가장 기본적인 '프레임 워크'이다.

## 딥러닝과 코드

딥러닝을 할 때 코드는 처음부터 다 짠다???

> 죽을 수도 있습니다

<br>

케라스 : wrapper라고 할 수 있음. 유저가 사용하기 편하도록 만들어 놓음. 하단에는 TensorFlow나 PyTorch를 선택할 수 있게 함. 비교적 high-level API

| 케라스            | 텐서플로             | 파이토치                   |
| ----------------- | -------------------- | -------------------------- |
| high-level API    | high & low level API | low-level API              |
| 간단한 구조. 쉬움 | 그렇게 쉽지는 않다   | 어려움                     |
| static graph      | static graph         | dynamic computation graphs |

</br>
<hr>

## Computation graph?

연산의 과정을 그래프로 표현한 것.

- Define and Run : 그래프를 먼저 정의 --> 실행 시점에서 데이터를 feed (TensorFlow)
- Define by Run : 실행을 하면서 그래프를 생성하는 방식. Dynamic computation graph, DCG

### Dynamic Graph

Static 대비 몇가지 장점이 존재한다. 훨씬 간결한 코드, debug의 유리함이 있다. static 환경에서는 그래프에 실제 값이 들어가야 확인이 가능하다.
</br></br>
![aa](/001.png)

static graph는 매 iteration에서 기존의 구축된 정적인 함수를 sess.run에 따라 값을 변화하며 사용한다. dynamic graph의 경우에는 매 iteration을 거치며 수정되는 파라미터를 통해 새로운 그래프를 생성한다.

## Why PyTorch

- Define by Run의 장점: 즉시 확인 가능 -> **pythonic code**
- GPU support, Good API
- 사용하기 편한 장점!
- Tensorflow는 production과 scalability에 장점이 있다.

pytorch는 Numpy 구조를 가지는 Tensor 객체로, array로 표현된다. 자동 미분(Auto Grad)을 지원하며 다양한 DL 연산을 지원한다. 또, DL을 지원하는 다양한 함수와 모델도 같이 지원한다.

> Auto Grad ?

<hr><hr>
</br>

# PyTorch Basics

> 상세한 코드 실습은 ipynb 파일 참조

```python
import numpy as np
import torch
n_array = np.arange(10).reshape(2,5)
print(n_array)
print("ndim":, n_array.ndim, " shape:", n_array.shape)

t_array = torch.FloatTensor(n_array)
print(t_array)
print("ndim":, t_array.ndim, " shape:", t_array.shape)
```

Tensor 생성은 list나 ndarray를 사용 가능하다. 그 외에도 기본적으로 tensor가 가질 수 있는 data 타입은 numpy와 동일하다.

## Tensor handling

view, squeeze, unsqueeze 등으로 tensor 조정 가능.

- view: reshape와 동일하게 shape를 변환
- squeeze: 차원의 개수가 1인 차원을 삭제 (압축)
- unsqueeze: 차원의 개수가 1인 차원을 추가

![](/002.png)

기본적인 tensor의 operation은 numpy와 동일하다. 단, 행렬곱셈 연산은 dot이 아닌 mm을 사용한다.

## nn.functional

해당 모듈을 통해 다양한 수식 변환을 지원한다.

## AutoGrad

PyTorch의 핵심은 자동 미분의 지원이다. -> Backward 함수

</br>
<hr>
<hr>

# PyTorch 프로젝트 구조의 이해

## ML 코드는 언제나 Juypter에서?

장점

- 초기 단계에서는 대화식 개발 과정이 좋다. 도출된 아이디어를 빠르게 구현해볼 수 있고, 학습과정과 디버깅 등에서 지속적인 확인이 가능하다.

한계점

- 배포 및 공유 단계에서는 notebook은 공유가 어렵다는 단점이 있다. 실행 순서가 중요하기 때문에 쉽게 재현하기 어렵다.
- DL 코드도 하나의 프로그램으로 개발 용이성도 중요하나, 유지보수의 향상도 필요하다.

### OOP + 모듈 -> 프로젝트

다양한 프로젝트 템플릿이 존재한다. 필요에 따라 수정하여 사용하면 된다. 실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등등 다양한 모듈들을 분리하여 프로젝트 템플릿화 하여 사용하자

- 추천 Template
- https://github.com/victoresque/pytorch-template

# Today's Todo

- Template 분석하고, 각 코드들이 어떤 역할을 하는지 주석으로 달아보기

### 느낀점

PyTorch 공식문서를 이렇게 많이 볼 줄은 몰랐다. 아마 태어나서 공식문서를 제일 오래 본 하루 아니었을까 싶다... 과제도 만만치 않은 양이었다. 특히 indexing과 slicing 관련 파트에서 많은 시간을 보냈다. 강의와 실습으로 배웠지만 여전히 약하구나 느꼈고, 이번 기회에 나름 제대로(?) 공부한 느낌이다. 관련해서 정리한 것도 곧 포스트로 올려보려고 한다.<br><br>
그외에도 문서에서 나름 중요한 것, 그럼에도 어느정도 세부적인 내용도 놓치지 않고 다 짚어주는데, 만드신 분이 정말 존경스러웠다. 특히, 공부를 하다보면 당연히 갖게 될 의문점도 제시해 주어서 좋았다. 예를 들면, Linear와 LazyLinear의 차이라던가, nn.Identity는 왜 쓰는지 등등<br><br>
여담으로 과제를 하면서 왜 후기들에서 부덕이(🦆)를 그렇게 외쳤는지 알게되었다... 하하.. 앞으로 몇주가 남았더라... 이제 2주차인데...
