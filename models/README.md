# Model ağırlıkları

Bu proje ağırlıkları EVREN platformunda açık kaynak olarak paylaşılmış altı eğitim koşusundan oluşur. Proje bağlantısı:

- https://evren.ssyz.org.tr/projects/01a1ffce-5161-7701-a3ac-c0b9d7102188

## Ağırlık listesi

| Model | Dosya / kaynak | Test mAP@50 |
|---|---|---:|
| `fruit_vegetables_veri2_v1` | EVREN proje sayfasındaki açık model kaydı; yerelde `models/fruit_vegetables_veri2_v1/best.pt` olarak kullanılabilir | 0.8677 |
| `fruit_vegetables_veri3_v1` | EVREN proje model kaydı | 0.7567 |
| `fruit_vegetables_veri4_v1` | EVREN proje model kaydı | 0.7930 |
| `fruit_vegetables_veri4_v2` | EVREN proje model kaydı | 0.8447 |
| `fruit_vegetables_veri4_v3` | EVREN proje model kaydı | 0.8417 |
| `fruits_vegetables_v4` | EVREN proje model kaydı | 0.8441 |

Ağırlıkların kanonik açık kaynak kopyası EVREN proje sayfasındadır. GitHub dosya yükleme kısıtları nedeniyle bu depoda ağırlık yerine model adı, deney sonuçları ve otomatik model keşif kodu tutulur.

Diğer modelleri EVREN'den indirip `models/<model_adı>/best.pt` konumuna koyduğunuzda Flask arayüzündeki model seçim alanında görünürler.
