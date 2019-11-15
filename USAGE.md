FILES
=====


prepare.py : train, test 를 하기 위한 이미지 파일을 준비하는 스크립트
train.py: Recycle GAN을 이용하여 모델을 훈련하는 스크립트
test.py: 훈련된 모델을 이용하여 이미지를 만드는 스크립트
mkvideo.py: 이미지를 모아 비디오로 만드는 스크립트

환경준비
=======
- X window가 되지 않는 환경이라면, USAGE-X-Window.md 를 참고하세요
- MacOS에서 iterm2를 사용하길 추천합니다. USAGE-Iterm2.md 를 참고하세요.
- 아나콘다 환경을 설정합니다
```
    . conda-horock
```

동영상 준비
===========
- youtube-dl을 설치합니다
```
  sudo apt install youtube-dl
```
- youtube-dl은 수시로 바뀌므로 최신 버전으로 업데이트합니다
```
  sudo youtube-dl --update
```
- URL을 다운로드합니다. ex) "muhyun.mkv", "jaein.mkv"
- 다운로드하기전 -F 옵션을 주어 포맷중 360p 정도되는 포맷을 선택합니다.
- 선택한 포맷 번호를 -f 에 주어 다운로드합니다.
```
 youtube-dl -F 'https://www.youtube.com/watch?v=4zg1oLQ3gOI'
 youtube-dl -f 133 -o muhyun.webm 'https://www.youtube.com/watch?v=4zg1oLQ3gOI'
 youtube-dl -F 'https://www.youtube.com/watch?v=GPKgbcO5ppw'
 youtube-dl -f 43 -o jaein.webm 'https://www.youtube.com/watch?v=GPKgbcO5ppw'
```

대표 사진 준비
==============
1) datasets/foo.mkv 파일이 있는지 확인합니다. 위의 예에서는 muhyun.mkv, jaein.mkv
2) 대표 이미지를 선택합니다.

    ./prepare.py pic foo

   위 명령은 foo.mp4, foo.mkv, foo.avi, foo.mov, foo.webm 파일을 차례로 찾아서 발견되는 동영상에 대해
   Full shot 으로 나온 화면을 임의 샘플링하여  datasets/foo/pic 디렉토리에 저장합니다.
   그리고, 랜덤하게 하나를 선택하여 datasets/foo.png 파일로 심볼릭 링크를 만들게 됩니다.
   원하지 않는 파일이면 수동으로 심볼릭 링크를 만들거나, 위 명령을 다시 실행하세요.
   실행중 'Error: Can't open display:' 오류가 나면 "환경준비"섹션을 확인하세요.

   만약 일부 특정 구간에서 추출하기를 원한다면, 예를 들어 시작 후 2000~3000번째 프레임에서 찾기를 원한다면 아래와 같이 합니다.

```
    ./prepare.py pic foo --begin=2000 --end=3000
```

   만약 샘플 수를 1000개에서 500개로 조정하고 싶다면

```
    ./prepare.py pic foo --samples=500
```

    임의 선택된 대표이미지만 다른 것으로 바꾸고 싶다면
```
    ./prepare.py pic foo --samples=0
```

    ex)
```
    ./prepare.py pic muhyun --begin=2000 --end=3000
    ./prepare.py pic jaein
```

트레이닝 영상 추출
==================
트레이닝을 위해 이미지를 추출합니다.

    ./prepare.py train foo

    위 명령은 foo.png 로 지정한 사람이 들어 있는 프레임들을 foo.mkv 에서 추출합니다. 추출한 파일들은

    ./datasets/foo/train

    에 저장됩니다.
    이때, train 디렉토리에는 AAAxBBB, CCCxDDD 형태로 크롭 영역의 크기로 된 이름의 서브디렉토리를 가지게 됩니다.

    ./datasets/foo/train/724x648
    ./datasets/foo/train/502x500

    이 서브디렉토리 중 5000개의 이미지가 먼저 쌓이면 추출을 중단합니다. (이 값은 --max=NNNN 로 바꿀 수 있습니다.)
    추출이 멈추고 나면 서브디렉토리 중 가장 많은 디렉토리가 존재하는 디렉토리를 images 라는 이름으로 심볼릭 링크를 겁니다.

    ./datasets/foo/images/

    위 디렉토리가 foo 인물을 트레이닝하기 위한 디렉토리로 사용됩니다.

1) 비디오 파일은 ./datasets/foo.mkv 를 사용하게 되나 강제로 바꾸려면 --video=datasets/foo2.mkv 옵션을 사용합니다.
2) 이미지 파일은 ./datasets/foo.png 를 사용하게 되나 강제로 바꾸려면 --picture=datasets/foo2.png 옵션을 사용합니다.
3) 트레이닝 진행되는 모니터링을 최소화하기 위해서는 --hide 옵션을 주어 끕니다.
4) 참조하는 프레임의 구간을 정하는 방법은 --begin=시작프레임 --end=종료프레임으로 지정합니다.
5) 모니터링 화면을 조절하기 위해서는 --scale=.5 와 같이 준다.
6) 얼굴 인식이 잘 안되는 경우 대표 이미지(foo.png)와 유사도차이 톨러런스를 조절합니다. (--reco=0.5)
    train 할 때 나오는 "Reco-dist: 0.18" 가 대표이미지와의 유사도입니다.
    이 값이 0에 가까울수록 유사한 인물입니다.
7) 원본 이미지는 내부적으로 가로 320pixel 이미지로 변환하여 인식을 시도합니다. (--detect_width=320)
    이 값이 너무 작으면 인식률이 떨어지고, 크면 처리 속도가 떨어집니다
8) full shot 을 판단하는 기준은 크롭영역의 크기가 전체 화면의 15%를 차지하는 경우를 말합니다. (--fulllshot=0.15)
9) 얼굴의 위치가 변하면 크롭영역이 달라지게 되는데, 새로운 영역이 기존 영역과 80% 겹치면 크롭영역을 유지합니다. (--overlap=.8)
    크롭 영역이 일치해야 같은 서브 디렉토리에 쌓이게 되므로, 얼굴 이동에 따라 적당히 조절합니다.
10) 장면전환은 가로길이 64픽셀의 HSV 모드로 변경한다음 이어지는 프레임간의 변화량의 평균값을 비교하여 
    80%이상이달라지면 장면이 전환됐음으로 인식하는데, 이 값은 --scene_threshold=70.0 으로 조절합니다.
    FADE-IN/FADE-OUT 혹은 OVERLAPPED-CHANGE가 일어나는 장면의 경우 장면 전환을 판단하기가 원칙적으로 쉽지 않습니다.
11) 얼굴과 배경이 현저한 색상차이가 있는 경우 배경을 제거하고 특정 색으로 채울 수 있습니다.

```
    ./prepare.py train muhyun --begin=2000 --end=3000
    ./prepare.py train jaein
```

테스트용 영상 추출
==================
train 영상과 달리 test 영상은 test1, test2, test3 과 같은 명령으로 만들어집니다.


    ./prepare test1 muhyun --begin=4000
    ./prepare test1 jaein --begin=3000

트레이닝
========

다음과 같이 training 합니다.

```
    ./train.py foo1 foo2
```

테스트
======

```
    ./test.py foo1 foo2


비디오합성
=========

```
    ./mkvideo.py foo1 foo2 AB

    or

    ./mkvideo.py foo1 foo2 AB
