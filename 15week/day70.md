# Office hour - MRC 코드 해설

Q-A Task

- Retriever
  - 주어진 쿼리와 유사한 문서들을 찾는다
- Reader
  - 받아온 유사한 문서 내에서 쿼리에 알맞은 답을 찾는다.

## Train ?

two-stage 모델이다.

- Reader 훈련
  - 관련 무서를 뽑아낸 상황에서 '정답'을 찾는 모델 훈련
- Retriever
  - 관련 문서를 뽑는 Retriever를 훈련
  - 사실 베이스라인의 TF-IDF는 Determinisitc 하므로 훈련 보다는 Fitting이 어울리는 표현. 따라서 이미 .pkl 파일이 있고, TF-IDF 설정이나 Passage가 바뀌지 않았다면 다시 수행하지 않아도 됨. (그래서 베이스라인에는 훈련 코드 없음)
  - 만약 필요하다면 직접 `Dense Passage Retrieval`을 추가해 학습해볼 수도 있다

### Retriever 훈련

- ppt 8페이지 참조

## train.py

Reader를 훈련

- train dataset에 대해서 학습이 되었음.
- 단순히 데이터셋에 저장된 context로부터 answer를 찾아낸다.
- Retrieval 과정은 포함 X reader만 성능 측정함
- 직접 파인튜닝한 모델의 결과가 있다면 --model_name_or_path로 불러오면 됨

## inference.py

wikipeida.json에서 데이터셋에 주어진 쿼리와 유사한 문서를 찾아서(Retrieval) 그 문서에서 쿼리에 맞는 답을 낸다. (Reader)

- 만약 dev 데이터로 Retriever + Reader 성능을 확인하려면 --do_eval
- 만약 test 데이터로 최종 결과를 내고 싶다면, --do_predict
