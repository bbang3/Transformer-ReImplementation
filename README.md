# Transformer-ReImplementation
[DMIS](https://dmis.korea.ac.kr/) 연구실에서 진행한 Transformer 모델 재구현 프로젝트입니다. ([Attention is All You Need (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html))

## Usage
### Train
```python
python main.py
```
### Inference
```python
python inference.py
```

## Dataset
IWSLT2017 en-de 데이터셋 (https://huggingface.co/datasets/iwslt2017)

## Settings
학습에 사용한 하이퍼파라미터는 다음과 같습니다.


|Name|Value|
|:-|-:|
|Epochs||
|Batch size|32|
|Learning rate||
|Dropout|0.1|

## Result

## References
원 논문 이외에 아래 레퍼런스를 추가로 참고하여 구현하였습니다.
- [Transformer (Attention Is All You Need) 구현하기](https://paul-hyun.github.io/transformer-01/)
- [Transformer를 이해하고 구현해보자!](https://kaya-dev.tistory.com/8)
