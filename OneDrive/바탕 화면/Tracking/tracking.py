import cv2 
import mediapipe as mp # 손가락 인식
import math
from PIL import ImageFont

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

                thumb_tip = hand_landmarks.landmark[4] # 엄지 

                index_tip = hand_landmarks.landmark[8] # 검지  

                middle_finger_tip = hand_landmarks.landmark[12] # 중지 

                ring_finger_tip = hand_landmarks.landmark[16] # 약지 
                ring_finger_pip = hand_landmarks.landmark[14] # 약지-2

                pinky_tip = hand_landmarks.landmark[20] # 소지 
                pinky_dip = hand_landmarks.landmark[19] # 소지-1
                
                dif1 = math.sqrt(((thumb_tip.x-index_tip.x)**2) + ((thumb_tip.y-index_tip.y)**2))
                # 엄지와 검지의 거리 
                dif2 = math.sqrt(((index_tip.x-middle_finger_tip.x)**2) + ((index_tip.y-middle_finger_tip.y)**2))
                # 검지와 중지의 거리 
                dif3 = math.sqrt(((middle_finger_tip.x-ring_finger_tip.x)**2) + ((middle_finger_tip.y-ring_finger_tip.y)**2))
                # 중지와 약지의 거리  
                dif4 = math.sqrt(((ring_finger_pip.x-pinky_dip.x)**2) + ((ring_finger_pip.y-pinky_dip.y)**2))
                # 약지와 소지의 거리  

                diff_result1 = float(dif1 * 500)
                diff_result2 = float(dif2 * 500)
                diff_result3 = float(dif3 * 500)
                diff_result4 = float(dif4 * 500)
                # 현재 가장 적절한 값은 500

                if diff_result1 <= 15.6:
                    cv2.putText(
                    image, text = '4-8', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                elif diff_result2 <= 25.9:
                    cv2.putText(
                    image, text = '8-12', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                elif diff_result3 <= 21.2:
                    cv2.putText(
                    image, text = '12-16', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                elif diff_result4 <= 17.3:
                    cv2.putText(
                    image, text = '16-20', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
                else :
                    cv2.putText(
                    image, text = '%d  %d  %d  %d' % (diff_result1, diff_result2, diff_result3, diff_result4) , org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)

                mp_drawing.draw_landmarks(
                    image,hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()