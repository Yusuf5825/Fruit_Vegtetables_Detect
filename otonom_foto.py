import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

# ==============================================================================
# AYARLAR — Buradan kendin düzenleyebilirsin
# ==============================================================================
KAYIT_KLASORU = Path(r"C:/Projeler/Fruits_detect/otonom_fotolar")
KAMERA_INDEX = 0          
ARALIK_SANIYE = 1.0       
TOPLAM_FOTO = 50        
GORUNTU_ONIZLEME = True   
UZANTI = ".jpg"


def benzersiz_id_uret() -> str:
    return uuid.uuid4().hex[:8]


def foto_adi_uret(uzanti: str = ".jpg") -> str:
    zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = benzersiz_id_uret()
    return f"image_{zaman_damgasi}_{uid}{uzanti}"


def kamera_ac(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera açılamadı (index={index}).")
    return cap


def main() -> None:
    if ARALIK_SANIYE <= 0:
        raise ValueError("ARALIK_SANIYE 0'dan büyük olmalıdır.")
    if TOPLAM_FOTO is not None and TOPLAM_FOTO <= 0:
        raise ValueError("TOPLAM_FOTO pozitif bir tam sayı veya None olmalıdır.")

    KAYIT_KLASORU.mkdir(parents=True, exist_ok=True)
    cap = kamera_ac(KAMERA_INDEX)

    cekilen = 0
    son_cekim_zamani = 0.0

    hedef_yazi = str(TOPLAM_FOTO) if TOPLAM_FOTO is not None else "Sınırsız"
    print(f"Başladı -> Klasör: {KAYIT_KLASORU} | Aralık: {ARALIK_SANIYE} sn | Hedef: {hedef_yazi}")
    if GORUNTU_ONIZLEME:
        print("Erken durdurmak için görüntü penceresinde 'q' tuşuna bas.")

    try:
        while True:
            if TOPLAM_FOTO is not None and cekilen >= TOPLAM_FOTO:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            simdi = time.monotonic()
            if cekilen == 0 or (simdi - son_cekim_zamani) >= ARALIK_SANIYE:
                dosya_adi = foto_adi_uret(UZANTI)
                tam_yol = KAYIT_KLASORU / dosya_adi
                if cv2.imwrite(str(tam_yol), frame):
                    cekilen += 1
                    son_cekim_zamani = simdi
                    print(f"[{cekilen}/{hedef_yazi}] Kaydedildi -> {dosya_adi}")

            if GORUNTU_ONIZLEME:
                cv2.imshow("Otonom Foto Cekim", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("Kullanıcı tarafından durduruldu.")
                    break
            else:
                # Önizleme kapalıysa CPU'yu yormamak için kısa bekleme
                time.sleep(0.02)

    finally:
        cap.release()
        if GORUNTU_ONIZLEME:
            cv2.destroyAllWindows()

    print(f"Tamamlandı. Toplam çekilen fotoğraf: {cekilen}")


if __name__ == "__main__":
    main()