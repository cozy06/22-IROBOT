import cv2
import mediapipe as mp
from playsound import playsound


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
        # print('face_landmarks:', face_landmarks)
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


count = 10
# face_video(count) #리소스 up
# face_detect_video(count) #리소스 down
for i in range(1, count + 1):
    globals()["landmarks-{}".format(i)] = face_img(f"capture/test-{i}.jpg", i)
print(globals()['landmarks-1'].landmark[0])
