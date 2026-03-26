# 🚀 Havacılıkta Yapay Zekâ Yarışması — Tam Çalıştırma Rehberi

Bu döküman, projedeki 3 ana görevin (Nesne Tespiti, Pozisyon Kestirimi, Görüntü Eşleme) nasıl çalıştırılacağını, test edileceğini ve konfigüre edileceğini anlatır.

---

## 📦 1. Kurulum ve Hazırlık

Tüm görevler için ortak bağımlılıkları yükleyin:

```bash
# Proje ana dizininde (cursor_gorev1)
pip install -r birinci_gorev/requirements.txt
pip install -r ikinci_gorev/requirements.txt
```

---

## 🔍 GÖREV 1: Nesne Tespiti (Object Detection)
**Dizin:** `birinci_gorev/`

Bu görev; araçları, insanları, park ve iniş alanlarını tespit eder.

### A. Yarışma Esnasında (Sunucu Mode)
Yarışma sunucusuna bağlanıp 2250 kareyi otomatik işlemek için:
```bash
python birinci_gorev/scripts/run_session.py --server http://SUNUCU_IP:PORT --token TAKIM_TOKEN_BURAYA
```

### B. Yerel Test (Local Mode)
Elinizdeki bir fotoğrafı test edip sonucunu görmek için:
```bash
python birinci_gorev/scripts/run_inference.py --image data/raw/rgb/test.jpg --visualize
```
*Çıktı şuraya kaydedilir:* `birinci_gorev/outputs/visualizations/`

---

## 📍 GÖREV 2: Pozisyon Tespiti (Position Estimation)
**Dizin:** `ikinci_gorev/`

Bu görev, GPS olmadığında sadece kamera ile (x, y, z) konumu hesaplar.

### A. Yarışma Esnasında
```bash
python ikinci_gorev/scripts/run_session.py --server http://SUNUCU_IP:PORT --token TAKIM_TOKEN_BURAYA
```

### B. Yerel Test
Bir klasör dolusu karenin pozisyonunu kestirip referanslarla karşılaştırmak için:
```bash
python ikinci_gorev/scripts/run_local_test.py --frames_dir data/raw/frames/ --positions_csv test_verisi.csv
```
*Not:* Eğer CSV yoksa, program ilk 450 kareyi "sağlıklı" kabul eder, sonrasını kendi hesaplar.

---

## 🖼️ GÖREV 3: Görüntü Eşleme (Image Matching)
**Dizin:** `ucuncu_gorev/`

Uçuş anında sunucudan gelen yeni nesneleri karelerde bulur.

### A. Yarışma Esnasında
```bash
python ucuncu_gorev/scripts/run_session.py --server http://SUNUCU_IP:PORT --token TAKIM_TOKEN_BURAYA
```

---

## 🛠️ Önemli Konfigürasyonlar (config.yaml)

Her görevin kendi içinde bir `config/config.yaml` dosyası bulunur. Kodun içine girmeden şu ayarları değiştirebilirsiniz:

- **Görev 1:** YOLO eşik değerleri (confidence), UAP/UAİ çapı (4.5m).
- **Görev 2:** Kamera odak uzaklığı (focal length), özellik dedektörü (SIFT/ORB).
- **Görev 3:** Çoklu-ölçek (multi-scale) ve Çoklu-açı (multi-angle) arama listeleri.

---

## 💡 İpuçları ve Notlar

1.  **Hız:** Eğer görev 3 çok yavaş gelirse, `config.yaml` içindeki `scales` (ölçekler) veya `angles` (açılar) listesini kısaltarak hızı artırabilirsiniz.
2.  **Hatalar:** Tüm hatalar her görevin kendi `outputs/` klasöründeki `.log` dosyalarına kaydedilir.
3.  **Model:** Görev 1'i çalıştırmadan önce `birinci_gorev/models/weights/` içine `best.pt` dosyanızı koyduğunuzdan emin olun.

---
*Hazırlayan: Antigravity AI Assistant*
