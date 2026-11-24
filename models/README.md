# Model ağırlıkları

Bu proje ağırlıkları EVREN platformunda açık kaynak olarak paylaşılmış altı eğitim koşusundan oluşur. Proje bağlantısı:

- https://evren.ssyz.org.tr/projects/01a1ffce-5161-7701-a3ac-c0b9d7102188

## Ağırlık listesi

| Model | Dosya / kaynak | Test mAP@50 |
|---|---|---:|
| `fruit_vegetables_veri2_v1` | `models/fruit_vegetables_veri2_v1/parts/` altında parçalı GitHub ağırlığı; Flask başlangıçta `best.pt` olarak birleştirir | 0.8677 |
| `fruit_vegetables_veri3_v1` | EVREN proje model kaydı | 0.7567 |
| `fruit_vegetables_veri4_v1` | EVREN proje model kaydı | 0.7930 |
| `fruit_vegetables_veri4_v2` | EVREN proje model kaydı | 0.8447 |
| `fruit_vegetables_veri4_v3` | EVREN proje model kaydı | 0.8417 |
| `fruits_vegetables_v4` | EVREN proje model kaydı | 0.8441 |

En iyi modelin parçaları, GitHub tek dosya yükleme sınırı nedeniyle `part-000.bin`, `part-001.bin` biçiminde tutulur. `web_flask/app.py` parçaları otomatik olarak `.model_cache/fruit_vegetables_veri2_v1/best.pt` dosyasına birleştirir. Bu dosyalar model ağırlığının parçalarıdır; doğrudan tek tek çalıştırılmaz.

Diğer modelleri EVREN'den indirip `models/<model_adı>/best.pt` konumuna koyduğunuzda Flask arayüzündeki model seçim alanında görünürler.
