from tkinter import E
import cv2 
import mediapipe as mp # 손가락 인식

from function import motion, print_num, get_dis

W = 500 # 가중치 

list = [] # 손이 이동한 거리를 저장하기 위한 리스트 
dis_temp = [0]
value = 0 

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

                # top
                thumb_tip = hand_landmarks.landmark[4] 
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]  
                ring_tip = hand_landmarks.landmark[16] 
                pinky_tip = hand_landmarks.landmark[20] 

                # 2 
                thumb_ip = hand_landmarks.landmark[3]
                index_dip = hand_landmarks.landmark[7]
                middle_dip = hand_landmarks.landmark[11]
                ring_dip = hand_landmarks.landmark[15]
                pinky_dip = hand_landmarks.landmark[19]

                # 3
                index_pip = hand_landmarks.landmark[6]
                middle_pip = hand_landmarks.landmark[10]
                ring_pip = hand_landmarks.landmark[14]
                pinky_pip = hand_landmarks.landmark[18]

                # 4 
                index_mcp = hand_landmarks.landmark[5]
                middle_mcp = hand_landmarks.landmark[9]
                ring_mcp = hand_landmarks.landmark[13]
                pinky_mcp = hand_landmarks.landmark[17]

                # 주먹 확인 코드 
                rock5 = get_dis(thumb_tip.x, index_pip.x, thumb_tip.y, index_pip.y, W) # 엄지 
                rock1 = get_dis(index_dip.x, index_mcp.x, index_dip.y, index_mcp.y, W) # 검지 
                rock2 = get_dis(middle_dip.x, middle_mcp.x, middle_dip.y, middle_mcp.y, W) # 중지
                rock3 = get_dis(ring_dip.x, ring_mcp.x, ring_dip.y, ring_mcp.y, W) # 약지 
                rock4 = get_dis(pinky_dip.x, pinky_mcp.x, pinky_dip.y, pinky_mcp.y, W) # 소지

                if ((((rock1 <= 2) and (rock2 <= 2)) and ((rock3 <= 2) and (rock4 <= 2))) and (rock5 <= 2)):
                    text = 'Click'
                    motion(image, text)
                    temp = index_mcp.x
                    temp = int(temp * 100)
                    list.append(temp)

                elif ((((rock1 <= 2) and (rock2 <= 2)) and ((rock3 <= 2) and (rock4 <= 2))) and (rock5 >= 2)):
                    text = 'GOOD'
                    motion(image, text)

                elif ((((rock1 <= 2) and (rock3 <= 2)) and ((rock4 <= 2) and (rock5 <= 2))) and (rock2 >= 2)):
                    text = 'Bad word'
                    motion(image, text)

                else:
                    print_num(image, rock5, rock1, rock2, rock3, rock4)
                    # value = list[0] - list[-1]
                    # print(f'이동한 거리 : {value}')
                    # dis_temp.append(value)
                    if len(list) > 0 :
                        value = list[0] - list[-1]
                    
                
                

                mp_drawing.draw_landmarks(
                    image,hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                

        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            print(list)
            print(value * -1)
            break

    cap.release()