import numpy as np
import cv2


lk_params = dict(winSize = (15, 15),
                           maxLevel = 2,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.4,
                      minDistance = 7,
                      blockSize = 7 )


trajectory_len = 20
detect_interval = 5
trajectories = []
frame_idx = 0


cap = cv2.VideoCapture("../video.mp4")
#cap = cv2.VideoCapture("../video1.mp4")

#Videoyu kaydetmeyi denedim ama olmadı.
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
#size = (frame_width, frame_height)
#video = cv2.VideoWriter("file.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
    
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    #Lucas Kanade Algoritmasını direkt olarak copy paste ettim.
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # En yeni tespit edilen nokta
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Tüm izlenen yolları çiz
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


    #Yeni özelliklerin güncellenmesi ve tespit edilmesi aralığı
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # İzlenen yolun (trajectory) son değeri 
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Takip edilecek iyi featurları belirleme
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # Eğer iyi bir feature ise izlenecek yollara (trajectories) eklenir.
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    #Bir sonraki frame'e geç ve şu anki frame'i eski frame'e ata.
    frame_idx += 1
    prev_gray = frame_gray

    # Görüntüleri ekranda göster
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    #Video yazdırma
    #video.write(img)

    #Kapatma ve anlık görüntüyü png olarak kayıt etmek için tuş ataması.
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    elif k == ord('s'):
        cv2.imwrite('img.png', img)
        cv2.imwrite('mask.png', mask)
    
    
cap.release()
#video.release()
cv2.destroyAllWindows()