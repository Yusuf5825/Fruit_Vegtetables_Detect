# Meyve & Sebze Tespit ve Sayım Sistemi (YOLOv11)

Bu depo; **Apple**, **Lemon**, **Potato**, **Tomato** ve **Carrot** sınıflarının bilgisayarlı görü ile tespiti ve sayımı amacıyla geliştirilen veri hazırlama, model eğitimi, gerçek test seti (`kendi_test_verim`) değerlendirmesi, canlı kamera testi ve Flask tabanlı web uygulaması (`web_flask`) bileşenlerini içerir[cite: 17, 18, 19, 27].

---

## 1. Sabit Sınıf Yapısı ve Depo Mimari Düzeni

Projede tüm veri birleştirme, etiket eşleme, değerlendirme ve uygulama adımlarında sabit 5 sınıf indeksi kullanılmıştır:

* `0`: **Apple**
* `1`: **Lemon**
* `2`: **Potato**
* `3`: **Tomato**
* `4`: **Carrot**[cite: 17]

### Depodaki Başlıca Klasör ve Dosyalar

* `web_flask/`: Flask tabanlı web uygulaması (`app.py`, `templates/index.html`, `static/css/style.css`, `static/js/app.js`); fotoğraf yükleme, kutu çizimi, sınıf sayımı ve model seçimi özellikleri sunar[cite: 27, 31].
* `test_sonuclari_json/`: Modellerin `kendi_test_verim` üzerinde değerlendirilmesiyle üretilen model bazlı JSON raporları ve `_tum_modeller_ozet.json` dosyası[cite: 18, 19, 20, 21, 22, 23, 24, 31].
* `Uygulama_test_icin_fotolar/`: Web arayüzü ve manuel test senaryoları için ayrılan örnek test görselleri[cite: 31].
* `data_merged.py` & `merge_dataset_v33.py`: Farklı veri kaynaklarını mevcut birleştirilmiş sürümlerin üzerine ekleme yapmadan yeni sürümler halinde tekilleştirerek (`SHA-256`) ve veri sızıntısı oluşturmadan birleştiren betikler[cite: 31].
* `split.py` & `kontrol.py`: Veri seti bölünme (`train` / `val` / `test`) ve eksik/boş etiket veya bütünlük kontrolleri için kullanılan betikler[cite: 31].
* `otonom_foto.py`: Belirlenen zaman aralığında (`ARALIK_SANIYE`) ve adette (`TOPLAM_FOTO`) `image_YYYYMMDD_HHMMSS_<benzersiz_id>.jpg` formatında otomatik fotoğraf çeken betik.
* `kamera_app.py`: Canlı kamera akışı üzerinde GPU ile nesne tespiti ve loglama sağlayan arayüzlü uygulama betiği[cite: 6, 31].
* `kendi_verim_test.py`: `models` altındaki `best.pt` ağırlıklarını `kendi_test_verim` setinde GPU (`cuda:0`) üzerinde test edip `mAP@50`, `mAP@50-95`, `Precision`, `Recall`, `F1` ve `Mean Matched IoU` değerlerini JSON olarak kaydeden değerlendirme betiği[cite: 18, 19, 31].

> **Veri Seti Saklama Notu:** Ham ve sıkıştırılmış (`.zip`) veri setleri boyutları ve sürüm geçmişi nedeniyle depoya doğrudan yüklenmemiştir; bunun yerine birleştirme, bölünme, doğrulama, test ve `run_manifest` kayıtları üzerinden tam izlenebilirlik sağlanmıştır[cite: 12, 31].

---

## 2. Veri Hazırlama, Sürümleme ve Negatif Örnek Serüveni

### Aşamalı Veri Birleştirme ve Sınıf Hizalama
* İlk birleştirme ve düzenleme adımlarında farklı kaynaklardan (`apple`, `carrot`, `isi312_zanahoria`, `lemon`, `Potato`, `tomato`, `coco_out`, `dataset` vb.) gelen etiketler hedef `0..4` sınıf sıralamasına hizalanmıştır[cite: 7, 9, 17, 27].
* Alt etiketleri yüzde/boyut değerleri (`'100'`, `'25'`, `'50'`, `'75'`) şeklinde tanımlanmış `apple.v4i.yolov8` gibi kaynaklarda ilgili etiketli kutular `0: Apple` sınıfına, tek sınıflı `carrot.v1i.yolov8` / `isi312_zanahoria` kaynaklarındaki etiketler ise `4: Carrot` sınıfına dönüştürülmüştür[cite: 7, 9, 10, 17].
* Yeni birleştirilmiş sürümler (`merged_dataset_v2`, `merged_dataset_v3`), önceki veri klasörlerinin üzerine doğrudan yazılmadan ayrı dizinlerde oluşturulmuş; içerik tabanlı **SHA-256** tekilleştirme uygulanarak aynı görüntünün farklı split'lere (`train`, `val`, `test`) sızması engellenmiştir.

### Negatif / Arka Plan Örnekleri ve Etiket Kalitesi
* Eğitim sürecinde platform istatistiklerinde görünen etiketsiz/negatif örnek oranları ile yerel diskteki `.txt` varlığı/boşluk kontrolleri (`kontrol.py` yaklaşımı) birlikte incelenmiş, `merged_dataset` üzerindeki etiket bütünlüğü doğrulanmıştır.
* Canlı kamerada hedef 5 sınıf dışında kalan nesnelere (yüz, kırmızı poşet, masa/oda nesneleri vb.) gereksiz kutu çizilmesini azaltmak için `otonom_foto.py` ile negatif/arka plan kareler toplanmış, boş etiketli (`.txt` içeriği boş) negatif örnekler `train` ve `val` split'lerine sızıntısız ayrılmış; ayrıca Pascal VOC gibi dış kaynaklardan da arka plan/iç mekan örnekleri çekme stratejisi kurgulanmıştır.

---

## 3. Model Eğitimi, Konfigürasyon ve `run_manifest` Yapısı

Eğitim koşularında Ultralytics (`yolo11l`) mimarisi ve `AdamW` optimizasyonu kullanılmıştır[cite: 12]. Örnek bir koşu manifestinden (`fruit_vegetables_veri4_v1` manifest kaydı) alınan temel konfigürasyon ve ortam parametreleri aşağıdadır[cite: 12]:

| Parametre / Alan | Değer / Açıklama |
| :--- | :--- |
| **Mimari & Ağırlık** | `yolo11l` (`base_weights: yolo11l.pt`, `task: detect`, `nc: 5`)[cite: 12, 17] |
| **Giriş Boyutu & Epoch** | `imgsz: 800`, `epochs: 200`, `patience: 50`[cite: 12] |
| **Batch & İşçi Sayısı** | `requested batch: 64`, `effective batch: 256` (`world_size: 4`), `workers: 8`[cite: 12] |
| **Optimizer & LR** | `optimizer: AdamW`, `lr0: 0.001`, `lrf: 0.01`, `momentum: 0.937`, `weight_decay: 0.0005`, `cos_lr: true`[cite: 12] |
| **Warmup & AMP** | `warmup_epochs: 3`, `warmup_momentum: 0.8`, `amp: true`, `half: true`[cite: 12] |
| **Kayıp Katsayıları** | `box: 7.5`, `cls: 0.5`, `dfl: 1.5`, `label_smoothing: 0`[cite: 12] |
| **Veri Artırma (Augmentation)** | `hsv_h: 0.01`, `hsv_s: 0.3`, `hsv_v: 0.2`, `degrees: 5`, `translate: 0.05`, `scale: 0.2`, `fliplr: 0.5`, `flipud: 0`, `erasing: 0.1`, `mosaic: 0` (veya sonraki denemelerde artırılmış `mosaic` / `scale` / `hsv` ayarları)[cite: 12] |
| **Örnek Split Sayısı** | Manifest kaydında `train` split'inde `25.405` görsel ve `sha256` split özeti yer almaktadır[cite: 12]. |

### `run_manifest` Dosyalarının Önemi
Her eğitim koşusu için üretilen `run_manifest_*.json` dosyaları; kullanılan mimariyi, istenen (`requested`) ve fiilen uygulanan (`effective`) hiperparametreleri, çalışma ortamı paket/donanım bilgilerini (`python`, `torch`, `ultralytics`, `cuda`, GPU) ve split dosya listeleri ile `sha256` özetlerini kayıt altına alır[cite: 12]. Böylece hangi ağırlığın hangi veri ve ayarlarla üretildiği sonradan geriye dönük olarak doğrulanabilir[cite: 12].

---

## 4. Normal / Eğitim Doğrulama Grafikleri ve `kendi_test_verim` Sonuçları

### Eğitim / Doğrulama Sürecindeki Gözlemler
* Eğitim/doğrulama grafiklerinde model ilerledikçe kayıp (`Loss`) düşmüş ve makro metrikler yükselmiştir (örneğin bir koşuda makro `mAP@50: %86.4`, `F1: %82.7`, `Precision: %85.0`, `Recall: %80.6` seviyesine ulaşmıştır)[cite: 11].
* Bununla birlikte sınıf bazlı eğrilerde bazı sınıfların diğerlerine göre daha düşük bantta kalması (örneğin bir eğride yeşil sınıf eğrisinin `~%58–60` bandında seyretmesi) ve patates-elma benzerliği ile havuçta dönemsel sapmalar gözlenmesi; veri çeşitliliği, sınıf dağılımı ve augmentasyon/güven eşiği ayarlarının iyileştirilmesi gerektiğini göstermiştir[cite: 5, 11].

### `kendi_test_verim` Değerlendirme Sonuçları (`test_sonuclari_json`)
Tüm modeller `kendi_test_verim` seti üzerinde `imgsz=640`, `conf=0.25`, `iou=0.50`, `cuda:0` ayarlarıyla test edilmiş ve çıktılar `test_sonuclari_json/` klasörüne kaydedilmiştir[cite: 18, 19, 20, 21, 22, 23, 24]:

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Mean Matched IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fruit_vegetables_veri2_v1` | **0.8677** | **0.5484** | 0.9173 | **0.8816** | **0.8967** | **0.7651** |
| `fruit_vegetables_veri4_v2` | 0.8447 | 0.5203 | 0.9207 | 0.8615 | 0.8878 | 0.7522 |
| `fruits_vegetables_v4` | 0.8441 | 0.5004 | 0.8907 | 0.8582 | 0.8657 | 0.7483 |
| `fruit_vegetables_veri4_v3` | 0.8417 | 0.5256 | 0.9186 | 0.8552 | 0.8851 | 0.7448 |
| `fruit_vegetables_veri4_v1` | 0.7930 | 0.4764 | 0.9033 | 0.8149 | 0.8442 | 0.7231 |
| `fruit_vegetables_veri3_v1` | 0.7567 | 0.4642 | **0.9311** | 0.7760 | 0.8369 | 0.6993 |

[cite: 18]

### En Yüksek Skorlu Model (`fruit_vegetables_veri2_v1`) Sınıf Bazlı Sonuçları
`kendi_test_verim` testinde en yüksek genel `mAP@50` (`0.8677`), `F1` (`0.8967`), `Recall` (`0.8816`) ve `Mean Matched IoU` (`0.7651`) değerini elde eden `fruit_vegetables_veri2_v1` modelinin sınıf bazlı performansı[cite: 18, 19]:

| Sınıf | Class ID | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Mean Matched IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple** | `0` | 0.8141 | 0.4943 | 0.8964 | 0.8278 | 0.8607 | 0.6775 |
| **Lemon** | `1` | 0.9076 | 0.4962 | 0.8428 | 0.9461 | 0.8915 | 0.7636 |
| **Potato** | `2` | 0.7647 | 0.5072 | 0.9310 | 0.7714 | 0.8438 | 0.8029 |
| **Tomato** | `3` | 0.9684 | 0.6463 | 0.9489 | 0.9766 | 0.9625 | 0.8259 |
| **Carrot** | `4` | 0.8837 | 0.5983 | 0.9676 | 0.8861 | 0.9251 | 0.7848 |

[cite: 19]

---

## 5. Canlı Kamera Testleri ve Web Uygulaması (`web_flask`)

### Canlı Kamera Testi Bulguları
* Gerçek zamanlı kamera denemelerinde aydınlatma koşulları, kamera mesafesi ve güven eşiği (`conf`) tespit kararlılığını doğrudan etkilemiştir.
* `0.35` güven eşiğinde bazı canlı `Lemon` ve `Carrot` örneklerinin kaçabildiği; eşik düşürüldüğünde (`0.20–0.25`) kaçırma azalırken, arka plandaki nesnelerde düşük güvenli yanlış pozitiflerin çıkabildiği gözlemlenmiştir.
* Ultralytics tarafında `half` parametresinin kullanım dışı kalmaya başlaması nedeniyle GPU çıkarım kodları güncel parametrelerle uyumlu hale getirilmiştir[cite: 4].

### Flask Web Uygulaması (`web_flask`)
* `fruit_vegetables_veri2_v1/best.pt` varsayılan ağırlığı ile **NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`)** üzerinde çalışacak şekilde yapılandırılmıştır[cite: 19, 27, 29].
* Kullanıcı arayüzü (`templates/index.html`, `static/css/style.css`, `static/js/app.js`) üzerinden fotoğraf sürükle-bırak veya seçme yoluyla yüklenebilir, güven eşiği (`conf`) ayarlanabilir, `models` klasöründeki farklı modeller seçilerek test edilebilir ve tespit edilen kutular ile sınıf bazlı adetler panelde görüntülenebilir[cite: 27, 31].

### Uygulamayı Çalıştırma
```bash
python -u web_flask/app.py
