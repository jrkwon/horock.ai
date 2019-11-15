iTerm2는 console 창에서 바로 image를 확인할 수 있는 터미널 확장이 있습니다.
- iTerm2 > Install Shell Integration 을 실행하거나 아래와 같이 실행합니다.
```
    curl -L https://iterm2.com/shell_integration/install_shell_integration_and_utilities.sh | bash
```
- 위 명령은 ~/.iterm2/ 디렉토리에 몇몇 스크립트를 설치합니다.
- 다시 로그인하거나 다음을 실행합니다.
```
    . ~/.iterm2_shell_integration.bash
```
- 이미지 파일을 확인하는 방법은 imgcat 이라는 명령을 사용합니다.
```
    imgcat datasets/muhyun.png
```
