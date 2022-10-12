import cv2 
import mediapipe as mp # 손가락 인식
import math
from PIL import ImageFont

def get_dis(dis):
    return float(dis * 500)

def print_text(str):
    cv2.putText(
    image, text = str, org=(10, 30),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
    color=255, thickness=2)

def print_num(dis_1, dis_2, dis_3, dis_4, dis_5):
    cv2.putText(
    image, text = '%d  %d  %d  %d  %d' % (dis_1, dis_2, dis_3, dis_4, dis_5) , org=(10, 30),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
    color=255, thickness=2)


mp_drawing = mp.solutions.drawing_utils # 웹캠 영상에서 뼈 마디를 인식함.
mp_hands = mp.solutions.hands # 동일하다 

cap = cv2.VideoCapture(0) # 웹 캠을 열어준다.

with mp_hands.Hands(
    max_num_hands=2, # 최대 손 인식 갯수
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
                # 2 랑 4 의 거리가 작아야 주먹이된다.

                dif1 = math.sqrt(((thumb_tip.x-index_tip.x)**2) + ((thumb_tip.y-index_tip.y)**2))
                # 엄지와 검지의 거리 
                dif2 = math.sqrt(((index_tip.x-middle_tip.x)**2) + ((index_tip.y-middle_tip.y)**2))
                # 검지와 중지의 거리 
                dif3 = math.sqrt(((middle_tip.x-ring_tip.x)**2) + ((middle_tip.y-ring_tip.y)**2))
                # 중지와 약지의 거리  
                dif4 = math.sqrt(((ring_pip.x-pinky_dip.x)**2) + ((ring_pip.y-pinky_dip.y)**2))
                # 약지와 소지의 거리  

                rock1 = (((index_dip.x-index_mcp.x)**2) + ((index_dip.y-index_mcp.y)**2))
                rock2 = (((middle_dip.x-middle_mcp.x)**2) + ((middle_dip.y-middle_mcp.y)**2))
                rock3 = (((ring_dip.x-ring_mcp.x)**2) + ((ring_dip.y-ring_mcp.y)**2))
                rock4 = (((pinky_dip.x-pinky_mcp.x)**2) + ((pinky_dip.y-pinky_mcp.y)**2))
                rock5 = (((thumb_tip.x-index_pip.x)**2) + ((thumb_tip.y-index_pip.y)**2))

                dis_1 = get_dis(rock1)
                dis_2 = get_dis(rock2)
                dis_3 = get_dis(rock3)
                dis_4 = get_dis(rock4)
                dis_5 = get_dis(rock5)

                # if dis_1 <= 15.6:
                #     print_text('4-8')
                # elif dis_2 <= 25.9:
                #     print_text('8-12')
                # elif dis_3 <= 21.2:
                #     print_text('12-16')
                # elif dis_4 <= 17.3:
                #     print_text('16-20')

                if dis_1 <= 2 and dis_2 <= 2 and dis_3 <= 2 and dis_4 <= 2 and dis_5 <= 2:
                    print_text('rock')
                elif dis_1 <= 2 and dis_2 <= 2 and dis_3 <= 2 and dis_4 <= 2 and dis_5 >= 2:
                    print_text('good')
                else:
                    print_num(dis_1,dis_2,dis_3,dis_4,dis_5)

                mp_drawing.draw_landmarks(
                    image,hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()