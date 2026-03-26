# ✈️ Havacılıkta Yapay Zekâ Yarışması — Nesne Tespiti

## 📌 Proje Özeti

Bu proje, **Havacılıkta Yapay Zekâ Yarışması** kapsamında hava aracının alt-görüş kamerasından elde edilen video karelerinde gerçek zamanlı nesne tespiti gerçekleştirmek amacıyla geliştirilmektedir.

## 🎯 Tespit Edilecek Sınıflar

| Sınıf         | ID | Hareket Durumu | İniş Durumu |
|---------------|----|----------------|-------------|
| Taşıt         | 0  | 0 (hareketsiz) / 1 (hareketli) | -1 |
| İnsan         | 1  | -1             | -1          |
| UAP           | 2  | -1             | 0 / 1       |
| UAİ           | 3  | -1             | 0 / 1       |

## 📁 Klasör Yapısı

```
cursor_gorev1/
├── config/          → Konfigürasyon dosyaları
├── data/            → Ham ve işlenmiş veriler
│   ├── raw/         → RGB & Termal ham görüntüler
│   ├── processed/   → İşlenmiş görüntüler
│   ├── annotations/ → Etiket dosyaları
│   └── splits/      → Train/Val/Test bölümleri
├── models/          → Model ağırlıkları ve konfigler
├── src/             → Kaynak kod
│   ├── detection/   → Tespit modülleri
│   ├── preprocessing/  → Ön işleme
│   ├── postprocessing/ → Son işleme (NMS vb.)
│   └── utils/       → Yardımcı araçlar
├── scripts/         → Çalıştırma betikleri
├── notebooks/       → Analiz notebook'ları
├── tests/           → Birim testler
├── outputs/         → Tahmin ve görsel çıktılar
└── docs/            → Proje dokümanları
```

## ⚡ Hızlı Başlangıç

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Tek görüntü üzerinde tahmin yap
python scripts/run_inference.py --image data/raw/rgb/ornek.jpg

# Toplu kare işleme
python scripts/run_batch.py --input_dir data/raw/rgb/ --output_dir outputs/predictions/
```

## 📋 Teknik Gereksinimler

- Video: Full HD veya 4K, 7.5 FPS, her oturum 2250 kare (5 dakika)
- Kamera: RGB veya Termal, alt-görüş, 70-90° açı
- Koşullar: Gündüz/gece, kar/yağmur, şehir/orman/deniz ortamları

## 📝 Önemli Kurallar

- Tren lokomotifi ve vagonları ayrı ayrı taşıt olarak etiketlenir
- Bisiklet/motosiklet sürücüsü → "insan" değil, taşıt+sürücü = taşıt
- Kısmen görünen nesneler de tespit edilmelidir
- Kamera hareketi ile nesne hareketi ayırt edilmelidir
