# Multi-GPU 학습

## Model Parallel
모델을 병렬화 하는 방법. gpu들이 모델의 각 부분을 병렬적으로 처리한다. 단점으로는 병목현상과 파이프라인 설계의 어려움으로 상당한 난이도를 가진다는 것이다.

## Data parallel
데이터를 나눠서 GPU에 할당 후 결과의 평균을 취한느 방법이다. mini-batch와 유사한 개념이며, 한번에 여러 gpu에서 병렬적으로 처리한다.
PyTorch 에서는 DataParallel, DistributedDataParallel이 있다.

- DataParallel - 단순히 데이터를 분배하고 평균을 구함. GPU 사용 불균형 문제와 병목 문제가 있을 수 있음
- DistributedDataParallel - 각 cpu마다 process를 생성하여 개별 gpu에 할당 후 처리

<br>
<hr>

# Hyperparameter Tuning
인공지능 모델이 스스로 학습하지 않는 여러 parameter를 직접 조정하는 과정이다. learning rate나 모델의 크기, optimzer 등등을 hyper parameter라고 부른다. 더 높은 정확도를 위해 마지막으로 시도해보는 일종의 튜닝 과정이라고 생각하자.

더 높은 정확도를 위해서는 일반적으로 다른 모델을 적용해보는 방법이 있다. 그 다음으로는 다른 데이터로 학습을 시도해보는 방법이 있다. Hyper parameter tunning은 가장 마지막으로 시도해보는 방법이다. 그만큼 성능 향상의 폭은 그렇게 크지는 않다.

## Grid and Random
Grid layout: 값의 범위를 일정하게 정하고 차례대로 골라서 학습을 시도해보는 방법. 예를 들어 batch-size를 32, 64, 128, ... 이렇게 정하고 차례대로 시도해본다.

Random layout: 실제 hyper parameter는 다양하기 때문에 (축이 큼) random search를 통해 적당히 grid search를 시도해볼 곳을 찾아낸다.

`최근에는 베이지안 기법을 주로 사용`

### Ray
- multi-node multi-processing 지원 모듈
- ML/DL 병렬 처리를 위해 개발됨
- 현재 분산 병렬 ML/DL 모듈의 표준임
- Hyperparameter Search를 위한 다양한 모듈도 제공

</br>

```python
data_dir = os.path.abspath('./data')
load_data(data_dir)

config = {
    'l1': tune.sample_from(lambda _:2**np.random.randint(2,9),
    'l2': tune.sample_from(lambda _:2**np.random.randint(2,9),
    'lr': tune.loguniform(1e-4, 1e-1),
    'batch_size':tune.choice([2, 4, 8, 16])
    } # config에 search space를 지정함

secheduler = ASHAScheduler( # ASHA 학습 스케줄링 알고리즘
# 학습 중간중간 큰 변화가 없는 hyper parameter들은 잘라내는 알고리즘
    metric='loss', mode='min', max_t=max_num_epoch, grace_period=1, reduction_factor=2)

reporter = CLIReporter( # 결과 출력 양식 지정
    metric_colums=['loss', 'accuracy', 'training_iteration'])

reulst = tune.run( # 병렬처리 방식으로 학습 실행
    partial(train_cifar, data_dir=data_dir),
    resources_per_trial={'cpu':2, 'gpu':gpus_per_trial},
    config=config, num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter
)
```

> 그래서 Hyper parameter tunning을 해? 말아?

현재는 그렇게 큰 효과를 보기 힘들다. 시간대비 효율도 떨어진다. 학습을 개선할 다른 방법들을 모두 시도해보고 최종적으로 도전해보는 방향으로 접근하자!

<br>
<hr>
<hr>

# PyTorch Troubleshooting
파이토치 환경에서 자주 발생하는 문제 및 해결방법

## OOM
out of memory...

메모리 문제는 어디서, 왜 발생했는지 알기 어렵다. Error backtracking을 하여도 최상위 코드(라이브러리에 있거나, 혹은 내가 짠 코드와는 아주 머나먼 곳...)으로 가기도 한다. 또, 컴퓨터의 메모리 상황을 파악하는 것은 꽤 어려운 작업이기도 하다.

> 간단하게 시도 할만 한 해결책

Bacht size를 줄여보자! GPU clean과 메모리 부담을 줄여보는 것으로 간단하게 해결 가능할 수도 있다.

## GPUUtil 사용
nvidia-smi 처럼 GPU의 상태 및 정보를 보여주는 모듈. Colab 환경에서 현재 GPU 환경을 확인하는데 큰 도움이 된다. 매 iter마다의 변화도 확인할 수 있음

```python
!pip install GPUtil

import GPUtil
GPUtil.showUtilization()
```

## cuda.empty_cache()
torch.cuda.empty_cache(): 사용되지 않은 cache를 정리하여 가용 메모리를 확보한다.

## trainning loop 과정 확인
trainning 과정에서는 tensor 변수를 사용하는데, tensor 변수는 메모리 사용량이 상당히 크다. 특히, 이러한 tensor가 적층되는 구조인 경우 GPU 메모리를 상당히 많이 차지하게 되며 문제가 될 수 있다. 이런 경우 1D tensor는 파이썬의 기본객체로 변환하여 저장하는 것이 좋다.

```python
for i in range(10000):
    optimizer.zero_grad()
    output = ...
    loss = criterion(output)
    loss.backward() # backward 진행
    optimizer.step()
    total_loss += loss # loss가 적체되며 문제 가능성 있음
```

```python
for x in range(10000):
    ...
    # .item으로 파이썬 기본객체로 변환하여 저장하자
    total_loss += loss.item
```

## del
필요가 없어진 변수는 삭제하는 것이 좋다. 파이썬 메모리 배치의 특성상 loop가 끝나도 메모리를 계속 차지하고 있기 때문이다.

```python
for x in range(10000):
    intermediate = f(input[x]) # for loop 밖에서도 접근 가능하다
    result += g(intermediate) # intermediate 사용이 끝나면 del을 하자
    del(intermediate)

output = h(result)
return output
```

## batch size 조절
batch size를 1로 해보고 시도해보자. batch size 문제가 아닐 수도 있기 때문이다. batch size 1이 문제없이 진행된다면 조금씩 늘려보자.

```python
oom = False
try:
    run_module(batch_size)
except RuntimeError: # Out of Memory
    oom = True

if oom:
    for _ in range(batch_size):
        run_module(1)
```

## torch.no_grad()
Inference 시점에서 no_grad()를 사용해보자. inference 과정에서 쓸데 없이 차지하는 메모리를 줄일 수 있음. backward 과정에서 쌓이는 메모리에서 자유롭다.

```python
with torch.no_grad():
    for data, target in test_loader:
        output = network(data)
        test_loss += F.null_loss(output, target, ...).item()
        ...
```

## 기타
CUDNN_STATUS_NOT_INT, device-side-assert등 cuda와 관련한 OOM 에러들이 몇가지 더 있다.

colab 환경에서는 지나치게 큰 모델은 사용할 수 없을 것이다. 예를 들자면, CNN이나 LSTM 같은 경우에는 colab에서는 실행시키기 힘들다.

CNN의 경우에는 OOM 보다는 torch.tensor의 크기가 맞지 않아 에러가 발생하는 경우가 많다. torchsummary 모듈을 사용하여 적절하게 현재 메모리 크기에 맞는 변수의 개수를 설정하여 사용하면 된다.

그 밖의 최적화 팁으로는 tensor의 float 표현을 16bit 짜리로 줄이는 것도 방법이다.