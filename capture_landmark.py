import cv2
import mediapipe as mp
from playsound import playsound
import pandas as pd


def fix_xyz(mark):
    standard = mark.landmark[164]
    chin = mark.landmark[18]
    ratio_x = 1 / chin.x
    ratio_y = 1 / chin.y
    ratio_z = 1 / chin.z
    for i in range(0, 468):
        mark.landmark[i].x = (mark.landmark[i].x - standard.x) * ratio_x
        mark.landmark[i].y = (mark.landmark[i].y - standard.y) * ratio_y
        mark.landmark[i].z = (mark.landmark[i].z - standard.z) * ratio_z
    return mark


def face_detect_video(num):
    img_count = 0
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("웹캠을 찾을 수 없습니다.")
                # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요.
                continue
            # 보기 편하기 위해 이미지를 좌우를 반전하고, BGR 이미지를 RGB로 변환합니다.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # 성능을 향상시키려면 이미지를 작성 여부를 False으로 설정하세요.
            image.flags.writeable = False
            results = face_detection.process(image)

            # 영상에 얼굴 감지 주석 그리기 기본값 : True.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) == ord('q'):
                playsound('sound/camera.mp3')
                img_count += 1
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.imwrite(f'capture/test-{img_count}.jpg', frame)  # 사진 저장
                if img_count == num:
                    break
    cap.release()
    cv2.destroyAllWindows()


def face_video(num):
    img_count = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("웹캠을 찾을 수 없습니다.")
                # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
                continue

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # 이미지 위에 얼굴 그물망 주석을 그립니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            # 보기 편하게 이미지를 좌우 반전합니다.
            cv2.imshow('MediaPipe Face Mesh(Puleugo)', cv2.flip(image, 1))
            if cv2.waitKey(5) == ord('q'):
                playsound('sound/camera.mp3')
                img_count += 1
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.imwrite(f'capture/test-{img_count}.jpg', frame)  # 사진 저장
                if img_count == num:
                    break
    cap.release()
    cv2.destroyAllWindows()


# 이미지 읽기
def face_img(img, i):
    mp_drawing_styles = mp.solutions.drawing_styles

    # 얼굴 검출을 위한 객체
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        static_image_mode=True,
        max_num_faces=1,
    )
    # Face Mesh를 그리기 위한 객체
    mp_drawing = mp.solutions.drawing_utils

    annotated_image = cv2.imread(img)

    # 얼굴 검출
    results = face_mesh.process(annotated_image)
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    cv2.imwrite(f"result/face-mesh-{i}.jpg", annotated_image)
    return face_landmarks


def distance(landmark):
    num = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    num[0][0] = landmark[164].x #인중
    num[0][1] = landmark[164].y
    num[0][2] = landmark[164].z
    num[1][0] = landmark[4].x #코
    num[1][1] = landmark[4].y
    num[1][2] = landmark[4].z
    num[2][0] = landmark[129].x
    num[2][1] = landmark[129].y
    num[2][2] = landmark[129].z
    num[3][0] = landmark[358].x
    num[3][1] = landmark[358].y
    num[3][2] = landmark[358].z
    num[4][0] = landmark[168].x #인중
    num[4][1] = landmark[168].y
    num[4][2] = landmark[168].z
    num[5][0] = landmark[33].x #눈
    num[5][1] = landmark[33].y
    num[5][2] = landmark[33].z
    num[6][0] = landmark[133].x
    num[6][1] = landmark[133].y
    num[6][2] = landmark[133].z
    num[7][0] = landmark[263].x
    num[7][1] = landmark[263].y
    num[7][2] = landmark[263].z
    num[8][0] = landmark[362].x
    num[8][1] = landmark[362].y
    num[8][2] = landmark[362].z
    num[9][0] = landmark[70].x # 눈썹
    num[9][1] = landmark[70].y
    num[9][2] = landmark[70].z
    num[10][0] = landmark[107].x
    num[10][1] = landmark[107].y
    num[10][2] = landmark[107].z
    num[11][0] = landmark[300].x
    num[11][1] = landmark[300].y
    num[11][2] = landmark[300].z
    num[12][0] = landmark[336].x
    num[12][1] = landmark[336].y
    num[12][2] = landmark[336].z
    num[13][0] = landmark[61].x #입
    num[13][1] = landmark[61].y
    num[13][2] = landmark[61].z
    num[14][0] = landmark[291].x
    num[14][1] = landmark[291].y
    num[14][2] = landmark[291].z
    num[15][0] = landmark[18].x
    num[15][1] = landmark[18].y
    num[15][2] = landmark[18].z
    num[16][0] = landmark[175].x #턱
    num[16][1] = landmark[175].y
    num[16][2] = landmark[175].z
    num[17][0] = landmark[367].x
    num[17][1] = landmark[367].y
    num[17][2] = landmark[367].z
    num[18][0] = landmark[135].x
    num[18][1] = landmark[135].y
    num[18][2] = landmark[135].z
    dis = list(range(0,171))
    j = 0
    for i in range(0,18):
        for a in range(i+1,19):
            dis[j] = ((abs(num[i][0]-num[a][0])**2) + (abs(num[i][1]-num[a][1])**2) + (abs(num[i][2]-num[a][2])**2))*1000000
            j += 1
    return dis

choose = int(input("학습/분류: "))
print(choose)
if choose is 1:
    name = int(input("학번: "))
    count = 5
    # face_video(count) #리소스 up
    face_detect_video(count)  # 리소스 down
    for i in range(1, count + 1):
        globals()["landmarks-{}".format(i)] = face_img(f"capture/test-{i}.jpg", i)

    Dis1 = distance(fix_xyz(globals()['landmarks-1']).landmark)
    Dis2 = distance(fix_xyz(globals()['landmarks-2']).landmark)
    Dis3 = distance(fix_xyz(globals()['landmarks-3']).landmark)
    Dis4 = distance(fix_xyz(globals()['landmarks-4']).landmark)
    Dis5 = distance(fix_xyz(globals()['landmarks-5']).landmark)
    Dis1.append(f"{name}_1")
    Dis2.append(f"{name}_2")
    Dis3.append(f"{name}_3")
    Dis4.append(f"{name}_4")
    Dis5.append(f"{name}_5")
    df = pd.DataFrame({'0': Dis1, \
                        '1': Dis2, \
                        '2': Dis3, \
                        '3': Dis4, \
                        '4': Dis5})
    df = df.transpose()
    print(df)
    df.to_csv('TF/TrainData.csv')
elif choose is 2:
    print("sdssdsdsd")
else:
    print("sdsd")
