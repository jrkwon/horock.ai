만약 사용하는 환경이 MacOS이고, Ubuntu 서버에 접속한다면,

- MacOS에 X-Quartz를 설치합니다.
- https://support.apple.com/ko-kr/HT201341
- X-Quartz를 실행한다음
- 평소 사용하는 터미널앱이나 iterm2를 이용하여 ubuntu를 다음과 같이 접속합니다.
```
ssh -X <hostname or ip>
```
- Ubuntu에서 DISPLAY 환경변수가 설정되어있는지 확인하고
```
$ env | grep DISPLAY
DISPLAY=localhost:10.0
```
- xcalc, xlogo, xeyes, xterm 등의 명령을 실행하여 새 창이 실행되는지 테스트해 봅니다.
