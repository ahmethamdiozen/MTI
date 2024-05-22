import cv2

# Kamerayı başlat
kamera = cv2.VideoCapture("../video.mp4")
#kamera = cv2.VideoCapture("../video1.mp4")

# İlk kareyi yakala
_, onceki_kare = kamera.read()

# Gri tonlama yap
onceki_kare_gri = cv2.cvtColor(onceki_kare, cv2.COLOR_BGR2GRAY)

# İki kare arasındaki hareket eşik değeri
hareket_esik_degeri = 50

while True:
    # Yeni kareyi yakala
    _, yeni_kare = kamera.read()

    # Gri tonlama yap
    yeni_kare_gri = cv2.cvtColor(yeni_kare, cv2.COLOR_BGR2GRAY)

    # İki kare arasındaki farkı hesapla
    fark = cv2.absdiff(onceki_kare_gri, yeni_kare_gri)

    # Hareket eşik değerini aşan farklı pikselleri belirle
    hareket_tespit = cv2.threshold(fark, hareket_esik_degeri, 255, cv2.THRESH_BINARY)[1]

    # Hareket tespiti için konturları bul
    konturlar, _ = cv2.findContours(hareket_tespit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hareket_algılandı = False

    # Konturları çiz
    for kontur in konturlar:
        if cv2.contourArea(kontur) > 50:  # Kontur alanı belirli bir değerin üstünde ise
            (x, y, w, h) = cv2.boundingRect(kontur)
            cv2.rectangle(yeni_kare, (x, y), (x + w, y + h), (0, 255, 0), 2)
            hareket_algılandı = True

    # Yeni kareyi göster
    cv2.imshow('Hareket Algılama', yeni_kare)

    # Hareket algılandığında veya algılanmadığında konsola yazı yaz
    if hareket_algılandı:
        print("Hareket var")
    else:
        print("Durgun")

    # Çıkış tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Bir sonraki iterasyon için kareleri güncelle
    onceki_kare_gri = yeni_kare_gri

# Kamerayı serbest bırak ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()