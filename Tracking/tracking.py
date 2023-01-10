import numpy as np
import cv2 
import mediapipe as mp # 손가락 인식

from function import motion, get_dis

W = 500 # 가중치 

move_x = [] # 손이 이동한 거리를 저장하기 위한 리스트 ( x축 )
move_y = [] # ( y축 )

value_x = 0

value_y = 0

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

                # tip
                thumb_tip = hand_landmarks.landmark[4] 
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]  
                ring_tip = hand_landmarks.landmark[16] 
                pinky_tip = hand_landmarks.landmark[20] 

                # dip
                thumb_ip = hand_landmarks.landmark[3]
                index_dip = hand_landmarks.landmark[7]
                middle_dip = hand_landmarks.landmark[11]
                ring_dip = hand_landmarks.landmark[15]
                pinky_dip = hand_landmarks.landmark[19]

                # pip
                index_pip = hand_landmarks.landmark[6]
                middle_pip = hand_landmarks.landmark[10]
                ring_pip = hand_landmarks.landmark[14]
                pinky_pip = hand_landmarks.landmark[18]

                # mcp
                index_mcp = hand_landmarks.landmark[5]
                middle_mcp = hand_landmarks.landmark[9]
                ring_mcp = hand_landmarks.landmark[13]
                pinky_mcp = hand_landmarks.landmark[17]
                
                # thumb_tip, index_pip
                IsRock = []
                t1 = get_dis(thumb_tip.x, index_pip.x, thumb_tip.y, index_pip.y, W)
                t2 = get_dis(ring_pip.x, middle_pip.x, ring_pip.y, middle_pip.y, W)
                IsRock.append([t1, t2])
                sum = np.sum(IsRock)

                if (sum <= 3.6688):
                    text = 'Click'
                    motion(image, text)

                    Xtemp = index_mcp.x
                    Xtemp = int(Xtemp * 100)
                    move_x.append(Xtemp)

                    Ytemp = index_mcp.y
                    Ytemp = int(Ytemp * 100)
                    move_y.append(Ytemp)

                else:
                    if value_x < -1 and (-5 <= value_y <= 5) :
                        text = 'right'
                        motion(image, text)
                    elif value_x > 1 and (-5 <= value_y <= 5) :
                        text = 'left'
                        motion(image, text)

                    elif value_y < -1 and (-5 <= value_x <= 5) :
                        text = 'down'
                        motion(image, text)

                    elif value_y > 1 and (-5 <= value_x <= 5) :
                        text = 'up'
                        motion(image, text)

                    elif value_x < -1 and value_y > 1 :
                        text = 'right up'
                        motion(image, text)

                    elif value_x < -1 and value_y < -1 :
                        text = 'right down'
                        motion(image, text)

                    elif value_x > 1 and value_y < -1 :
                        text = 'left down'
                        motion(image, text)

                    elif value_x > 1 and value_y > 1 :
                        text = 'left up'
                        motion(image, text) 
                    else :
                        text = 'Waiting'
                        motion(image, text)
        
                    
                    if len(move_x) > 0 :
                        value_x = (move_x[0] - move_x[-1])

                    if len(move_y) > 0 :
                        value_y = (move_y[0] - move_y[-1])

                    move_x.clear()
                    move_y.clear()

                mp_drawing.draw_landmarks(
                     image,hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
