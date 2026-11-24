# Meyve ve Sebze Tespit Sistemi

YOLO11 ile fotoğraflarda Apple, Lemon, Potato, Tomato ve Carrot nesnelerini bulan, sınıflara göre sayan bilgisayarlı görü projesi. Depoda veri hazırlama betikleri, `kendi_test_verim` değerlendirme kayıtları ve Flask arayüzü bulunur. İnceleme için sağlanan yedi manifest dosyası altı farklı eğitim konfigürasyonunu içerir.

> **En iyi ölçülen model:** `fruit_vegetables_veri2_v1` — `kendi_test_verim` üzerinde mAP@50 **0.8677**, mAP@50–95 **0.5484**, F1 **0.8967**. Değerlendirme `imgsz=640`, `conf=0.25`, `IoU=0.50` ayarlarıyla yapılmıştır.
>
> **Çalıştırılabilirlik notu:** Model ağırlıkları (`best.pt`) bu GitHub deposunda bulunmuyor. Bu nedenle uygulama, model dosyası projeye eklenmeden veya `MODEL_DIR` ile bir konuma yönlendirilmeden gerçek tahmin üretemez. Depoda şu anda yayımlanmış bir çevrimiçi demo bağlantısı da yoktur.

## Proje özeti

- **Görev:** Nesne tespiti ve nesne sayımı.
- **Sınıflar:** `0 Apple`, `1 Lemon`, `2 Potato`, `3 Tomato`, `4 Carrot`.
- **Model:** Ultralytics YOLO11; koşuların çoğu `yolo11l`, iki koşu `yolo11m`.
- **Veri düzeni:** Her eğitim koşusu için train/validation/test ayrımı ve SHA-256 özetleri incelenen manifestlerde kaydedilmiştir. Kendi test kümesi eğitim ve doğrulamadan ayrı tutulmuştur.
- **Uygulama:** `web_flask/` içinde fotoğraf yükleme, güven eşiği seçme, tespit kutularını ve sınıf adetlerini görüntüleme arayüzü.

## Hızlı gezinme

| İçerik | Konum |
|---|---|
| Ödev raporu | [`RAPOR.md`](RAPOR.md) |
| Flask uygulaması | [`web_flask/`](web_flask/) |
| Kişisel test sonuçları | [`test_sonuclari_json/`](test_sonuclari_json/) |
| Uygulama için fotoğraflar | [`Uygulama_test_icin_fotolar/`](Uygulama_test_icin_fotolar/) |
| Veri birleştirme | [`data_merged.py`](data_merged.py), [`merge_dataset_v33.py`](merge_dataset_v33.py) |
| Split ve etiket kontrolü | [`split.py`](split.py), [`kontrol.py`](kontrol.py) |
| Kendi test setinde değerlendirme | [`kendi_verim_test.py`](kendi_verim_test.py) |
| Kamera uygulaması ve fotoğraf toplama | [`kamera_app.py`](kamera_app.py), [`otonom_foto.py`](otonom_foto.py) |

## Test sonuçları

Modeller aynı `kendi_test_verim` değerlendirme kümesinde karşılaştırılmıştır. Tablo, repodaki `_tum_modeller_ozet.json` dosyasından alınmıştır.

| Model | mAP@50 | mAP@50–95 | Precision | Recall | F1 | Eşleşen kutuların ort. IoU |
|---|---:|---:|---:|---:|---:|---:|
| **fruit_vegetables_veri2_v1** | **0.8677** | **0.5484** | 0.9173 | **0.8816** | **0.8967** | **0.7651** |
| fruit_vegetables_veri4_v2 | 0.8447 | 0.5203 | 0.9207 | 0.8615 | 0.8878 | 0.7522 |
| fruits_vegetables_v4 | 0.8441 | 0.5004 | 0.8907 | 0.8582 | 0.8657 | 0.7483 |
| fruit_vegetables_veri4_v3 | 0.8417 | 0.5256 | 0.9186 | 0.8552 | 0.8851 | 0.7448 |
| fruit_vegetables_veri4_v1 | 0.7930 | 0.4764 | 0.9033 | 0.8149 | 0.8442 | 0.7231 |
| fruit_vegetables_veri3_v1 | 0.7567 | 0.4642 | **0.9311** | 0.7760 | 0.8369 | 0.6993 |

En iyi modelin sınıf bazlı F1 değerleri: Apple **0.8607**, Lemon **0.8915**, Potato **0.8438**, Tomato **0.9625**, Carrot **0.9251**. Tomato en yüksek F1'ı, Potato ve Apple ise geliştirmeye en açık sınıf sonuçlarını vermiştir.

## Eğitim koşuları ve model konfigürasyonları

Aşağıdaki değerler incelenen `run_manifest` dosyalarındaki **effective** ayarlardır. `batch` ve `world_size` koşular arasında değiştiği için her modelin ayarı ayrı gösterilmiştir. Ortak ayarlar: `imgsz=800`, `lr0=0.001`, `lrf=0.01`, `AdamW`, `momentum=0.937`, `weight_decay` tabloda, cosine learning-rate schedule, `warmup_epochs=3`, `AMP=true`, `box=7.5`, `cls=0.5`, `dfl=1.5`, `seed=0`.

| Koşu | Mimari | Görsel sayısı train/val/test | Epoch / patience | Batch / world size | Weight decay | Augmentation `hsv(h,s,v)`, scale, derece, translate, mosaic |
|---|---|---:|---:|---:|---:|---|
| fruit_vegetables_veri2_v1 | YOLO11l | 16,218 / 3,243 / 2,162 | 200 / 50 | 32 / 1 | 0.0005 | 0.015, 0.70, 0.40; 0.50; 10; 0.10; 1.0 |
| fruit_vegetables_veri3_v1 | YOLO11m | 17,888 / 3,577 / 2,384 | 200 / 50 | 512 / 8 | 0.0005 | 0.010, 0.30, 0.20; 0.20; 5; 0.05; 0 |
| fruit_vegetables_veri4_v1 | YOLO11l | 25,405 / 3,162 / 1,990 | 200 / 50 | 256 / 4 | 0.0005 | 0.010, 0.30, 0.20; 0.20; 5; 0.05; 0 |
| fruit_vegetables_veri4_v2 | YOLO11l | 25,405 / 3,162 / 1,990 | 200 / 50 | 128 / 2 | 0.001 | 0.010, 0.45, 0.30; 0.35; 10; 0.10; 0.6 |
| fruit_vegetables_veri4_v3 | YOLO11l | 26,605 / 3,462 / 1,990 | 200 / 35 | 384 / 6 | 0.001 | 0.010, 0.45, 0.30; 0.35; 10; 0.10; 0.6 |
| fruits_vegetables_v4 | YOLO11l | 3,104 / 388 / 388 | 150 / 35 | 48 / 1 | 0.0005 | 0.015, 0.70, 0.40; 0.50; 0; 0.10; 1.0 |

Seven manifest files were supplied for review; two `fruit_vegetables_veri3_v1` files are identical, so the configuration table lists six distinct runs. The manifest files themselves are not committed in this repository. İncelenen GPU eğitim manifestlerinde NVIDIA H200, CUDA 12.4, Python 3.10.12 ve Ultralytics 8.3.0 kayıtlıdır. These are the training environment details, not a hardware requirement for inference.

## Loss, output and evaluation

YOLO11's detection head predicts bounding boxes and class confidence scores. Training uses the Ultralytics detection loss with box regression, classification and Distribution Focal Loss components; the recorded loss weights are `box=7.5`, `cls=0.5`, `dfl=1.5`. The detection head uses sigmoid probabilities for class/object confidence and decodes box coordinates; it does not use a single softmax class output because multiple objects can appear in one image.

The project reports Precision, Recall, F1, mAP@50, mAP@50–95 and mean matched IoU. The best run's recorded evaluation duration is **89.27 seconds** for the full evaluation job. This is not the time for one image; per-image inference latency is measured by the Flask app and depends on the device.

## Flask uygulamasını çalıştırma

### Windows

1. Python 3.10 veya üzerini kurun.
2. Gerekli Python paketlerini bir sanal ortamda kurun: `pip install -r web_flask/requirements.txt`.
3. En iyi ağırlığı şu konuma koyun: `models/fruit_vegetables_veri2_v1/best.pt`.
4. `python web_flask/app.py` komutunu çalıştırıp `http://127.0.0.1:5000` adresini açın.

### Linux / macOS

Aynı adımları terminalde uygulayın. GPU yoksa uygulama CPU'ya döner; yanıt daha yavaş olabilir.

Uygulama `.jpg`, `.jpeg`, `.png`, `.webp` ve `.bmp` alır. Güven eşiği aralığı `0.05–0.95`, varsayılan değer `0.35`; çıkarım görsel boyutu `640` pikseldir. Diğer model ağırlıkları aynı yapıda `models/<model_adi>/best.pt` klasörlerine konursa arayüzde seçilebilir.

> Flask uygulamasını başlatmak için Python ortamı ve ağırlık dosyası gerekir. Eğitmen için tarayıcıdan, kurulum yapmadan açılan çevrimiçi demo henüz yayımlanmış değildir. Bunun için model ağırlıklarının herkese açık ya da dağıtıma aktarılabilir olması ve bir barındırma hedefi gerekir.

## Veri kaynağı ve yeniden üretilebilirlik

Veri birleştirme betikleri farklı kaynaklardan gelen etiketleri beş sınıfa eşler; dosya isimleri ve hash kayıtları split tekrarını izlemeye yardımcı olur. Veri ağırlıkları repo boyutu nedeniyle eklenmemiştir. İnceleme için sağlanan manifestler split sayıları ve SHA-256 özetlerini içerir. Ancak manifestlerde kaynak veri kümelerinin kalıcı URL'leri bulunmadığından, kullanılan her upstream veri kaynağının doğrudan URL'si bu repo içeriğinden doğrulanamıyor. Rapor içinde veri kaynakları ve erişim bilgisi bu nedenle şeffaf biçimde not edilmiştir.

## Ödev isterleri kontrolü

| İster | Bu repodaki kanıt |
|---|---|
| Problem ve uygun model | Nesne tespiti, YOLO11, 5 sınıf |
| Veri ve train/validation/test | Koşu manifestleri ve split sayıları; veri kaynağı URL'si eksik |
| Hiperparametreler, optimizer ve loss | Bu README'deki koşu matrisi ve `RAPOR.md` |
| Metrikler | `test_sonuclari_json/` JSON raporları |
| Eğitim ortamı ve süre | Manifestler; test değerlendirme süresi kaydı var. Eğitim toplam süresi manifestlerde yok |
| Çalışan deployment | Flask kaynak kodu mevcut; `best.pt` repo dışında, halka açık demo yok |
| 5 telefon fotoğrafı, görsel sonuç ve gerçek etiketle kıyas | Fotoğraf klasörü mevcut; beş görüntü için ground truth/tahmin karşılaştırma tablosu rapora eklenmeli |

## Repository

[GitHub deposu](https://github.com/Yusuf5825/Fruit_Vegtetables_Detect)
