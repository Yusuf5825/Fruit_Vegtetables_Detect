# Meyve ve Sebze Tespiti ve Sayımı Proje Raporu

## Özet

Bu projede tek bir fotoğraftaki Apple, Lemon, Potato, Tomato ve Carrot nesnelerinin konumları YOLO11 ile bulunmuş ve sınıflara göre sayılmıştır. Farklı veri sürümleriyle eğitilen modeller aynı kişisel test kümesinde karşılaştırılmıştır. En yüksek genel sonuç `fruit_vegetables_veri2_v1` koşusunda elde edilmiştir: mAP@50 0.8677, F1 0.8967 ve ortalama eşleşen kutu IoU'su 0.7651. Model, fotoğraf yükleme ve tahmin sonuçlarını kullanıcıya gösterme amacıyla Flask uygulamasına bağlanmıştır.

## 1. Proje bilgileri

| Alan | Yanıt |
|---|---|
| Proje başlığı | Meyve ve Sebze Tespit ve Sayım Sistemi |
| Problem / görev | Görüntüde hedef meyve ve sebzelerin nesne tespiti; kutu, sınıf ve adet üretimi |
| Sınıflar | Apple, Lemon, Potato, Tomato, Carrot |
| Veri kümesi | Birden fazla açık kaynaklı meyve-sebze veri kümesinden birleştirilmiş ve etiketleri beş hedef sınıfa eşlenmiş veri; ayrıca bağımsız `kendi_test_verim` kümesi |
| Veri kümesi URL | Kullanılan upstream kaynakların kalıcı URL'leri mevcut run manifestlerinde kayıtlı değil. Kaynak adları ve indirme URL'leri eklenmeden bu alan tam olarak doğrulanamıyor. |
| Repository | https://github.com/Yusuf5825/Fruit_Vegtetables_Detect |

## 2. Veri hazırlama

Farklı kaynaklardaki etiketler beş sınıflı hedef sıraya çevrilmiştir: Apple=0, Lemon=1, Potato=2, Tomato=3, Carrot=4. Görüntüler train, validation ve test kümelerine ayrılmıştır. Her eğitim manifesti split sayılarını ve SHA-256 özetlerini tutar. Bu hash kayıtları içerik tekrarını ve split'ler arası olası sızıntıyı kontrol etmek için kullanılır.

En iyi model koşusunun manifestinde 16.218 train, 3.243 validation ve 2.162 test görüntüsü vardır. `kendi_test_verim` ayrıca model karşılaştırmasında kullanılmıştır; değerlendirme betiği `imgsz=640`, `conf=0.25` ve IoU eşiği `0.50` ayarlarını kaydeder. Test raporunda kişisel test veri kümesinin görüntü sayısı açıkça verilmediğinden bu sayı burada belirtilmemiştir.

## 3. Model ve eğitim yöntemi

Mimari Ultralytics YOLO11 nesne tespitidir. Çoğu koşuda `yolo11l`, iki koşuda `yolo11m` temel ağırlığı kullanılmıştır. Girdi boyutu eğitimde 800 piksel, özel test değerlendirmesinde 640 pikseldir. YOLO tespit başlığı bir görüntüde birden fazla nesne bulunduğunu varsayar; her tespit için kutu koordinatlarını, sınıf skorlarını ve güven puanını üretir. Sınıf güven skorları sigmoid tabanlıdır; kutu koordinatları tespit başlığının regresyon çıktısından elde edilir.

Ultralytics eğitimindeki bileşik tespit kaybı kutu regresyonu, sınıflandırma ve Distribution Focal Loss bileşenlerinden oluşur. Manifestlerde kaydedilmiş katsayılar sırasıyla `box=7.5`, `cls=0.5` ve `dfl=1.5` değerleridir. Bu seçim, kutu konumunu ve sınıfı birlikte optimize ederken kutu dağılımını daha ayrıntılı öğrenmeyi hedefler.

### Ortak hiperparametreler

| Parametre | Değer |
|---|---:|
| Girdi boyutu | 800 (eğitim), 640 (kişisel test) |
| Maksimum epoch | Çoğu koşuda 200; `fruits_vegetables_v4` koşusunda 150 |
| Optimizer | AdamW |
| Başlangıç öğrenme oranı | 0.001 |
| Son öğrenme oranı oranı | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 veya 0.001 (koşuya göre) |
| LR schedule | Cosine (`cos_lr=true`) |
| Warmup | 3 epoch |
| AMP | Açık |
| Kayıp katsayıları | box=7.5, cls=0.5, dfl=1.5 |
| Diğer artırmalar | yatay çevirme 0.5; dikey çevirme 0; erasing 0.1; mixup 0; shear 0; copy-paste 0 |

### Deney konfigürasyonları

`batch` değerleri manifestlerdeki fiilen kullanılan (`effective`) değerlerdir. Görüntü sayıları her koşunun kendi split'ine aittir.

| Koşu | Model | Train / val / test | Epoch / patience | Batch / GPU sayısı | hsv(h,s,v) | scale / derece / translate / mosaic |
|---|---|---:|---:|---:|---|---|
| `fruit_vegetables_veri2_v1` | YOLO11l | 16,218 / 3,243 / 2,162 | 200 / 50 | 32 / 1 | 0.015 / 0.70 / 0.40 | 0.50 / 10 / 0.10 / 1.0 |
| `fruit_vegetables_veri3_v1` | YOLO11m | 17,888 / 3,577 / 2,384 | 200 / 50 | 512 / 8 | 0.010 / 0.30 / 0.20 | 0.20 / 5 / 0.05 / 0 |
| `fruit_vegetables_veri4_v1` | YOLO11l | 25,405 / 3,162 / 1,990 | 200 / 50 | 256 / 4 | 0.010 / 0.30 / 0.20 | 0.20 / 5 / 0.05 / 0 |
| `fruit_vegetables_veri4_v2` | YOLO11l | 25,405 / 3,162 / 1,990 | 200 / 50 | 128 / 2 | 0.010 / 0.45 / 0.30 | 0.35 / 10 / 0.10 / 0.6 |
| `fruit_vegetables_veri4_v3` | YOLO11l | 26,605 / 3,462 / 1,990 | 200 / 35 | 384 / 6 | 0.010 / 0.45 / 0.30 | 0.35 / 10 / 0.10 / 0.6 |
| `fruits_vegetables_v4` | YOLO11l | 3,104 / 388 / 388 | 150 / 35 | 48 / 1 | 0.015 / 0.70 / 0.40 | 0.50 / 0 / 0.10 / 1.0 |

### Eğitim ortamı

Manifestlerde eğitim ortamı Python 3.10.12, Ultralytics 8.3.0, CUDA 12.4 ve NVIDIA H200 olarak kayıtlıdır. Dağıtık koşular 1, 2, 4, 6 veya 8 GPU kullanmıştır. Bu, eğitim altyapısını tanımlar; Flask uygulamasının CPU'da tahmin yapması da mümkündür, ancak daha yavaş çalışır.

Manifest dosyaları eğitim toplam süresini içermemektedir. Bu nedenle eğitim süresi için doğrulanmış bir sayı raporlanamıyor. En iyi koşunun `kendi_test_verim` üzerindeki tam değerlendirme işi 89.27 saniye sürmüştür; bu değer tek görüntünün çıkarım süresi değildir.

## 4. Performans değerlendirmesi

Nesne tespiti için Precision, Recall, F1, mAP@50, mAP@50–95 ve eşleşen tahmin kutularının ortalama IoU metriği raporlanmıştır. `mAP@50`, IoU eşiği 0.50 iken ortalama hassasiyeti; `mAP@50–95`, 0.50 ile 0.95 arasındaki IoU eşiklerinde ortalama hassasiyeti ölçer. Precision yanlış pozitifleri, Recall kaçırılan gerçek nesneleri, F1 ise ikisi arasındaki dengeyi gösterir.

### Kişisel test kümesinde model karşılaştırması

| Model | mAP@50 | mAP@50–95 | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|
| **fruit_vegetables_veri2_v1** | **0.8677** | **0.5484** | 0.9173 | **0.8816** | **0.8967** | **0.7651** |
| fruit_vegetables_veri4_v2 | 0.8447 | 0.5203 | 0.9207 | 0.8615 | 0.8878 | 0.7522 |
| fruits_vegetables_v4 | 0.8441 | 0.5004 | 0.8907 | 0.8582 | 0.8657 | 0.7483 |
| fruit_vegetables_veri4_v3 | 0.8417 | 0.5256 | 0.9186 | 0.8552 | 0.8851 | 0.7448 |
| fruit_vegetables_veri4_v1 | 0.7930 | 0.4764 | 0.9033 | 0.8149 | 0.8442 | 0.7231 |
| fruit_vegetables_veri3_v1 | 0.7567 | 0.4642 | **0.9311** | 0.7760 | 0.8369 | 0.6993 |

### En iyi modelin sınıf bazlı metrikleri

| Sınıf | mAP@50 | mAP@50–95 | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|
| Apple | 0.8141 | 0.4943 | 0.8964 | 0.8278 | 0.8607 | 0.6775 |
| Lemon | 0.9076 | 0.4962 | 0.8428 | 0.9461 | 0.8915 | 0.7636 |
| Potato | 0.7647 | 0.5072 | 0.9310 | 0.7714 | 0.8438 | 0.8029 |
| Tomato | 0.9684 | 0.6463 | 0.9489 | 0.9766 | 0.9625 | 0.8259 |
| Carrot | 0.8837 | 0.5983 | 0.9676 | 0.8861 | 0.9251 | 0.7848 |

`fruit_vegetables_veri2_v1`, en iyi mAP@50, mAP@50–95, Recall, F1 ve mean matched IoU değerlerini vermiştir. `fruit_vegetables_veri3_v1` Precision değeri 0.9311 ile daha yüksektir; bu nedenle tek bir metrik tüm kullanım tercihlerini temsil etmez. Tomato en yüksek F1 değerine sahipken Potato'nun mAP@50 ve F1 değerleri beş sınıf içinde en düşüktür. Bu fark, sınıf örneklerinin çeşitlendirilmesi ve benzer görünümlü örneklerin artırılması için bir iyileştirme alanı gösterir.

## 5. Beş telefon fotoğrafıyla gerçek görüntü testi

Ödev yönergesi beş telefon fotoğrafı, model çıktısının görsel sunumu ve her fotoğraftaki tahminlerin gerçek etiketlerle karşılaştırılmasını ister. Depoda uygulama testi için fotoğraflar vardır; ancak bu görseller için gerçek nesne sayıları ve tahmin çıktıları eşlenmiş bir sonuç tablosu veya anotasyon dosyası bulunmamaktadır. Uygulama ayrıca `best.pt` ağırlığı olmadan çalıştırılamadığı için bu beş görsel için doğrulanmış tahmin ekran görüntüleri üretilememiştir.

Teslimden önce aşağıdaki tabloyu beş farklı telefon fotoğrafından çıkan gerçek uygulama sonuçlarıyla tamamlamak gerekir. Gerçek etiketleri görsel incelemeyle sayın; tahmin sütunlarını uygulamanın çıktısından yazın ve her satıra anotasyonlu görsel ekleyin.

| Fotoğraf | Gerçek sayım (Apple/Lemon/Potato/Tomato/Carrot) | Model tahmini | Doğru / eksik / fazla tespit | Görsel çıktı |
|---|---|---|---|---|
| 1 | Eklenmeli | Eklenmeli | Eklenmeli | Eklenmeli |
| 2 | Eklenmeli | Eklenmeli | Eklenmeli | Eklenmeli |
| 3 | Eklenmeli | Eklenmeli | Eklenmeli | Eklenmeli |
| 4 | Eklenmeli | Eklenmeli | Eklenmeli | Eklenmeli |
| 5 | Eklenmeli | Eklenmeli | Eklenmeli | Eklenmeli |

Bu gözlemler tamamlandığında başarılı ve başarısız örnekler; ışık, uzaklık, örtüşme, arka plan ve nesne boyutuna göre yorumlanmalıdır. Sayısal test kümesi metrikleri, bu beş fotoğrafın bireysel sonuçlarının yerine geçmez.

## 6. Deployment ve uygulama

Flask uygulaması kullanıcıdan fotoğraf alır, YOLO ile tahmin çalıştırır, kutular çizilmiş görseli, sınıf adetlerini, seçilen model adını ve ölçülen çıkarım süresini döndürür. Güven eşiği kullanıcı tarafından ayarlanabilir. Kaynak kodu `web_flask/app.py` ve ön yüz dosyaları `web_flask/templates/` ile `web_flask/static/` altındadır.

Bu anki GitHub sürümünde model dosyaları yoktur ve herkese açık çevrimiçi demo adresi yayımlanmamıştır. Dolayısıyla hocanın bağlantıya tıklayıp tarayıcıda doğrudan tahmin yapabilmesi için uygulamanın bir sunucuya dağıtılması ve model ağırlığının da o sunucuda erişilebilir olması gerekir. Modeli barındırma/dağıtma konumu belirlenip ağırlık eklendiğinde çevrimiçi demo bağlantısı bu bölüme yazılmalıdır.

Yerel çalıştırma için Python ve `web_flask/requirements.txt` paketleri gereklidir. Bu proje raporunun tek tıkla tarayıcı deneme ihtiyacını karşılaması için gelecekteki deployment, bu yerel kurulumdan ayrı olarak yapılmalıdır.

## 7. Sonuç

Proje beş sınıflı nesne tespiti problemi için YOLO11 tabanlı bir çözüm, birden fazla model deneyi, ayrılmış kişisel test değerlendirmesi ve Flask uygulama kodu sunmaktadır. Kişisel test sonuçlarında en iyi genel denge `fruit_vegetables_veri2_v1` modelindedir. Son teslimden önce rapordaki eksik kanıtlar tamamlanmalıdır: kaynak veri kümelerinin doğrudan URL'leri, eğitim toplam süresi, beş telefon fotoğrafının tahmin/gerçek etiket karşılaştırması ve ağırlık dosyasının erişilebilir olduğu canlı demo bağlantısı.

## Kaynaklar

1. Ultralytics. YOLO11 dokümantasyonu ve model görevleri. https://docs.ultralytics.com/models/yolo11/
2. Ultralytics. Model doğrulama ve mAP metrikleri. https://docs.ultralytics.com/modes/val/
3. Proje GitHub deposu ve ekli `run_manifest` / test JSON kayıtları. https://github.com/Yusuf5825/Fruit_Vegtetables_Detect
4. Veri kaynaklarının kalıcı URL'leri proje manifestlerinde bulunmadığından, upstream veri sayfaları kayıt altına alındığında bu kaynakça güncellenmelidir.
