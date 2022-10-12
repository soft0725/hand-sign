import math

# list_x = [1,4]
# list_y = [5,2]

# dif = math.sqrt(((list_x[0]-list_x[1])**2) + ((list_y[0]-list_y[1])**2))

# print(dif)
# print(type(dif))


import cv2 
import mediapipe as mp # 손가락 인식
import math

mp_drawing = mp.solutions.drawing_utils # 웹캠 영상에서 뼈 마디를 인식함.
mp_hands = mp.solutions.hands # 동일하다 

cap = cv2.VideoCapture(0) # 웹 캠을 열어준다.

with mp_hands.Hands(
    max_num_hands=1, # 최대 손 인식 갯수
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read() # 1 프레임씩 읽어온다.
        if not success: # 읽어오지 못하면 다음 프레임으로 넘어간다.
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # Opencv는 BGR을 사용하고, Mediapipe는 RGB를 사용한다. 
        # 그래서 COLOR_BGR2RGB로 바꿔주어야 하고, 
        # filp을 사용하여 거울 처럼 되어있는 것을 좌우를 바뀌어서 이미지를 받아준다.
        result = hands.process(image)
        # hands.process() 전처리 및 모델 추론을 함께 실행 

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks: # 이미지가 인식이 되어서 True가 된다면 
            for hand_landmarks in result.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4] # x y 값이 들어감 
                index = hand_landmarks.landmark[8] # x y 값이 들어감 
                # 엄지 손가락의 인덱스 = 4
                # 검지 손가락의 인덱스 = 8

                diff = math.sqrt(((thumb.x-index.x)**2) + ((thumb.y-index.y)**2))
                # 거리는 엄지와 검지의 x 좌표값의 거리 차이의 절대값을 이용 
                # diff 는 0 ~ 1 사이의 값을 가지게 된다. 
                # 그래서 아래와 같이 해줘야 함 

                diff_result = float(diff * 500)
                # 500 이라는 값은 수정 해줘야 할 수도 있다. 

                if diff_result <= 15.6 :
                    cv2.putText(
                    image, text = '0', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)

                elif diff_result >= 100:
                    cv2.putText(
                    image, text = '100', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                else :
                    cv2.putText(
                    image, text = '%d' % int(diff_result), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)


                mp_drawing.draw_landmarks(
                    image,hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()