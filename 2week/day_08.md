# 모델 불러오기

## model.save()
모델을 중간에 저장할 필요가 있다.
학습의 결과를 저장하기 위해 쓰며, 모델 형태와 파라미터를 저장한다. 모델 학습 동안 중간 과정을 저장해 두는 것으로 최선의 모델을 선택할 수 있다. 또, 만들어낸 모델을 공유할 수도 있음

- model.state_dict(): 모델의 파라미터를 표시
- model.load_state_dict(): 같은 모델 형태에서 파라미터만 load

### torchsummary
출력을 위한 유틸리티


## checkpoints
학습의 중간 결과를 저장하여 그 중에서 가장 최선의 모델을 선택하는 기법. earlystopping 기법을 사용하여 이전의 학습 결과물을 저장한다. 이때 loss와 metric 값을 지속적으로 확인 및 저장하여 선택에 활용한다.

```python
torch.save({
    'epoch': e,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    },
    f"saved/checkpoint_model_{e}_{epoch_loss/len(dataloader)}_{epoch_acc/len(dataloader)}.pt")
```

## Transfer learning
남이 만든 모델을 써보고 싶다.

다른 데이터셋으로 만든 모델을 현재 데이터에 적용하는 기법이다. 일반적으로 대용량 데이터 셋으로 만들어진 모델일 수록 성능이 좋기 때문이다. 현재 DL에서는 가장 일반적인 학습 기법이다. backbone architecture가 잘 학습된 모델을 가져와 일부분만 튜닝하며 학습을 수행시키는 방식이다.

> TorchVision은 다양한 기본 모델을 제공한다. NLP에서는 HuggingFace가 사실상의 표준 위치를 차지하고 있음.

## Freezing
pretrained model을 활용시 모델의 일부분은 적용하지 않는다. 이를 frozen 시킨다고 한다. 일부를 frozen 시킨 후 학습을 진행하면서 차츰 layer를 녹여 활용한다.

```python
vgg = models.vgg16(pretrained=True).to(device) # vgg16 모델을 가져와 vgg에 할당

class MyNewNet(nn.Module):
    def __init__(self):
        super(MyNewNet, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.linear_layers = nn.Linear(1000, 1) # 모델 마지막에 Linnear layer를 추가했다.

    def forward(self, x):
        x = self.vgg19(x)
        return self.linear_layers(x)

for param in my_model.parameters():
    param.requires_grad = False
# 마지막 레이어를 제외하고 frozen
for param in my_model.linear_layers.parameters():
    param.requires_grad = False

```
<hr>
<hr>
</br>

# Monitoring tools
머신러닝은 긴 학습 시간이 소요된다. 기다리는 시간이 상당하다. 간단하게는 print()나, log, csv등을 쓸 수 있지만, 학습 과정을 자동적으로 기록해주는 도구가 큰 도움이 될 수 있다.

## Tensorboard
- TensorFlow의 프로젝트로 만들어진 시각화 도구
- 학습 그래프, metric, 학습 결과의 시각화 지원
- PyTorch도 연결 가능

여러가지 값들을 저장할 수 있다.
- scalar : metric 등 상수 값의 연속
- graph: 모델의 computational graph
- histogram: weight 등 값의 분포를 표현
- image: 예측 값과 실제 값을 비교
- mesh: 3d 형태의 데이터를 표현

### 사용법
```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(logs_base_dir)


```

## weight & biases
- 머신러닝 실험을 지원하기 위한 상용도구
- 협업, code versioning, 실험결과 기록 등 제공
- MLOps의 대표적 툴로 발전중


<hr>
<hr>
</br>


# 과제 관련
## 정리해볼 함수들
- torch.gather()
- torch.nn.functional.pad()
- torch.register_backward_hook(): 과제 backward hook 3번째 코드 다시 보기