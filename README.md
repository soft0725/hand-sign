# Opencv를 활용하여 손의 움직임을 파악하는 프로그램입니다. 

### 프로젝트를 한 이유 : 굳이 마우스로만 페이지를 넘겨야 할까라는 생각에서 시작되었습니다. <br>

> ### 1. 손을 인식하는 방법 
[mediapipe 사이트](https://google.github.io/mediapipe/solutions/hands.html)
위 사이트에서 코드를 받아서 활용을 하였습니다.  <br><br>

![인덱스 번호](https://mediapipe.dev/images/mobile/hand_landmarks.png)

<br><br><br>

> ### 2. 인덱스의 거리 구하는 방법 
![점과 점 사이의 거리 공식을 할용](https://mblogthumb-phinf.pstatic.net/MjAxODAzMjdfMTg2/MDAxNTIyMTE0NDE3Mzkx.FGE2-XvEZMJ4gRvYEoikCTVUYIiFrs58nPKoI8n41U8g.gC8WPbkNO9zoWFkCigXKJ6gIQGNsdetzU7SXwzOUNGAg.PNG.tipsware/20180327_103323_023.png?type=w800)
<br>점과 점 사이의 공식을 활용하였습니다.  
왜인지는 모르겠지만 ((x1 - x2)^2 + (y1 - y2)^2)에 루트를 하지 않는게 더 코드가 복잡하지 않고 간편해서 공식을 제대로 활용하지는 않았습니다.

<br><br><br>

> ### 3. 사용 방법 
### 손을 펴고 있으면 Waiting 이라는 표시와 함께 사용자의 입력을 기다립니다.

![Waiting](./Users/a0102/OneDrive/%EB%B0%94%ED%83%95%20%ED%99%94%EB%A9%B4/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202022-10-14%20084213.png)

> ### 4. 사용 영상 
https://www.youtube.com/watch?v=B1fqY4H-08I
