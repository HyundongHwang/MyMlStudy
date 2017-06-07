# Neural Network Model을 학습시키면 어떻게 실세계와 유사해 지는가?

<!-- TOC -->

- [Neural Network Model을 학습시키면 어떻게 실세계와 유사해 지는가?](#neural-network-model을-학습시키면-어떻게-실세계와-유사해-지는가)
    - [개요](#개요)
    - [아이디어](#아이디어)
    - [준비](#준비)
    - [원본 함수](#원본-함수)
    - [실행과정](#실행과정)
    - [실행코드](#실행코드)
    - [실행결과](#실행결과)
    - [비교 플로팅](#비교-플로팅)

<!-- /TOC -->

<br>
<br>
<br>

## 개요
NN을 공부하다보니 문득 의문이 들었다. 
NN이 모델만 어느정도 복잡하게 두고 학습만 만땅 시키면 실세계를 모두 예측할 수 있다는데 정말 그럴까?
보통은 cost가 줄어드는것으로 대략적으로 이해했는데 실제 우리가 이해할 수 있는 2차원이나 3차원 plot으로 보면 좀 감이 올까?

## 아이디어
입력값 2개를 받아서 출력값 1개를 만들어내는 시스템이라면 3차원이라고 할 수 있고, 3차원이면 입력과 출력값이 인지할 수 있는 3차원 공간에 original plot을 그릴수 있다.
NN 트레이닝을 진행하는 과정에서 한개씩 plot을 그리면 어떤방식으로 최적화가 되는지 확인 할 수 있을것이다.

## 준비
- python 개발 환경
- 종속성 라이브러리 
    - numpy : 행렬연산등 각종 수학관련
    - tensorflow : 머신러닝
    - matplotlib : 2D plot
    - mpl_toolkits : 3D plot

## 원본 함수
<!--
y = 10{x_{0}}^{2} + 20{x_{1}}^{2}
-->
![](https://latex.codecogs.com/png.latex?%5Cbg_white%20y%20%3D%2010%7Bx_%7B0%7D%7D%5E%7B2%7D%20&plus;%2020%7Bx_%7B1%7D%7D%5E%7B2%7D)

## 실행과정
- 원본 함수 설정
- 원본 데이타 준비
    - -10 ~ 10 까지 원본함수를 이용한 입력/출력 정답 데이타 생성
- 원본 데이타 플로팅
- neural-network 구성
    - layer count : 4
    - hidden layer size : 20
    - activation function : relu
    - optimizer : Adam
- 트레이닝
- 스텝이 진행됨에 따라 cost 감소를 확인하고 이를 플로팅
- 원본 플로팅과 ML후 플로팅 결과를 비교함.

## 실행코드
- [/nn-proof.py](/nn-proof.py)

## 실행결과

```powershell
PS C:\project\170101_MyMlStudy\my-examples> python.exe .\nn-proof.py
step :  0
cost :  1.20204e+06
step :  1
cost :  915330.0
step :  2
cost :  536243.0
step :  3
cost :  297364.0
step :  10
cost :  211817.0
step :  20
cost :  94063.1
step :  30
cost :  64623.9
step :  100
cost :  1867.34
step :  200
cost :  711.1
step :  300
cost :  724.752
step :  999
cost :  172.36
```

## 비교 플로팅

<!--
0, 1, 2, 3, 10, 20, 30, 100, 200, 300, 999
-->

- 원본 데이타 플로팅

![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/original.png)

- ML 트레이팅 cost별 플로팅

![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(0).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(1).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(2).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(3).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(10).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(20).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(30).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(100).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(200).png)
![](https://hhdpublish.blob.core.windows.net/publish/nn-proof/ML%20result%20step(300).png)
