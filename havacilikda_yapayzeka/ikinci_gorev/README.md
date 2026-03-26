# ✈️ Havacılıkta Yapay Zekâ Yarışması — Görev 2: Pozisyon Tespiti

## 📌 Görev Özeti

Hava aracının konumlandırma sistemi (GPS vb.) kullanılamaz veya güvenilemez hale geldiğinde,
**yalnızca kamera görüntüleri** kullanılarak hava aracının **x, y, z** pozisyonunun kestirilmesi.

## 🎯 Hedef

Referans koordinat sisteminde başlangıç noktası `(0, 0, 0)` olarak verilir.
Ardışık kameradan gelen kareler ve ilk karelere ait referans pozisyon bilgisi kullanılarak
hava aracının metre cinsinden yer değiştirmesi hesaplanır.

## 📋 Oturum Yapısı

| Parametre            | Değer                          |
|----------------------|--------------------------------|
| Toplam süre          | 5 dakika                       |
| FPS                  | 7.5                            |
| Toplam kare          | 2250                           |
| İlk 1 dakika (450 kare) | Pozisyon bilgisi **sağlıklı** (kesin) |
| Son 4 dakika (1800 kare) | Pozisyon **sağlıksız** olabilir |
| Kamera açısı         | 70–90° (yere bakacak şekilde)  |

## 📡 Sunucudan Alınan Veriler (Her Kare İçin)

| Veri               | Açıklama                                                |
|--------------------|---------------------------------------------------------|
| Video Karesi       | Görüntü dosyası (jpg/png)                               |
| Pozisyon X (m)     | Referans X yer değiştirmesi                             |
| Pozisyon Y (m)     | Referans Y yer değiştirmesi                             |
| Pozisyon Z (m)     | Referans Z yer değiştirmesi (yükseklik)                 |
| Sağlık Değeri      | `1` = güvenilir pozisyon, `0` = güvenilmez → kendin hesapla |

## ⚡ Hızlı Başlangıç

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Yerel test (kaydedilmiş karelerle)
python scripts/run_local_test.py --frames_dir data/raw/frames/ --output_dir outputs/positions/

# Yarışma oturumu
python scripts/run_session.py --server http://HOST:PORT --token TEAM_TOKEN
```

## 📁 Klasör Yapısı

```
ikinci_gorev/
├── config/config.yaml          → Kamera parametreleri ve konfigürasyon
├── data/
│   ├── raw/frames/             → Ham görüntü kareleri
│   ├── processed/              → İşlenmiş veriler
│   └── calibration/            → Kamera kalibrasyon dosyaları
├── models/weights/             → Öğrenilmiş model ağırlıkları (opsiyonel)
├── src/
│   ├── position/               → Pozisyon kestirimi modülleri
│   ├── preprocessing/          → Görüntü ön işleme
│   ├── api/                    → Sunucu iletişimi
│   └── utils/                  → Yardımcı araçlar
├── scripts/                    → Çalıştırma betikleri
├── outputs/                    → Pozisyon ve görsel çıktılar
└── docs/                       → Dokümanlar
```
