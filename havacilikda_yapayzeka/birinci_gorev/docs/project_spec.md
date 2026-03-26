# Yarışma Şartnamesi Notları

## Video Teknik Bilgileri

| Parametre            | Değer                          |
|----------------------|--------------------------------|
| Süre / Oturum        | 5 dakika                       |
| FPS                  | 7.5                            |
| Kare sayısı          | 2250                           |
| Çözünürlük           | Full HD (1920×1080) veya 4K    |
| Format               | JPG, PNG veya herhangi bir format |
| Kamera açısı         | 70–90°                         |
| Kamera türü          | RGB veya Termal                |

## Zorlayıcı Durumlar

- **Yükseklik değişimi:** Kalkış, iniş ve seyrüsefer → farklı ölçekler
- **Hava koşulları:** Yağmur, kar vb.
- **Ortam:** Şehir, orman, deniz
- **Görüntü bozulması:** Bulanıklık, ölü piksel, donma, tam kayıp
- **Kısmi görünürlük:** Çerçeve dışı veya nesne arkasında kalan nesneler

## Kamera Açısı Uyarısı

- 70–90° arası (topdown'a yakın) → **uygun**
- 0–70° arası → uzaktaki nesneler gözden kaçabilir

## Puanlama Özeti (Genel)

- Her oturum için 2250 sonuç beklenmektedir
- Tespit edilen her nesne için: class_id, bbox, motion_status, landing_status bilgileri iletilmelidir

---

## Algoritma Çalışma Şartları (2.1.3)

| Kural | Açıklama |
|-------|----------|
| Sıralı kare | Bir kareye sonuç gönderilmeden sonraki kare istenemez |
| Toplu indirme yasak | Kareler yalnızca tek tek alınabilir |
| 1 sonuç / kare | İlk gönderilen sonuç değerlendirmeye alınır |
| Fazla gönderim | Aşım durumunda gönderim kabiliyeti geçici olarak engellenebilir |

### Bağlantı Akışı

```
[Bağlan] → [Kare İste] → [Tahmin Yap] → [Sonuç Gönder] → [Kare İste] → ...
                                ↑                              |
                                └──────────────────────────────┘
```

### Sunucu İstemci

Tam pipeline script'i: `scripts/run_session.py`

```bash
python scripts/run_session.py --server http://HOST:PORT --token TEAM_TOKEN
```
