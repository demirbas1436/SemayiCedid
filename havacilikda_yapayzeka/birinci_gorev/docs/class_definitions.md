# Sınıf Tanımları — Nesne Tespiti

## Tespit Edilecek 4 Sınıf

### Taşıt (ID: 0)

**Hareket Durumu:** `0` = Hareketsiz | `1` = Hareketli  
**İniş Durumu:** `-1` (Geçersiz)

**Kapsam:**
- Motorlu karayolu taşıtları: otomobil, motosiklet, otobüs, kamyon, traktör, ATV
- Raylı taşıtlar: tren, lokomotif, vagon, tramvay, monoray, füniküler
- Tüm deniz taşıtları

**Önemli Kurallar:**
- Tren lokomotifi ve vagonları ayrı ayrı birer taşıt olarak etiketlenir.
- Bisiklet/motosiklet sürücüsü **insan olarak etiketlenmez** → taşıt + sürücü = sadece **taşıt**
- Scooter: sürücüsüz → **taşıt** | sürücülü → **insan**
- Kamera hareketi ile gerçek taşıt hareketi ayırt edilmelidir.

---

### İnsan (ID: 1)

**Hareket Durumu:** `-1` (N/A)  
**İniş Durumu:** `-1` (N/A)

Ayakta veya oturan fark etmeksizin tüm insanlar. Kısmen görünenler de dahil.

---

### UAP — Uçan Araba Park (ID: 2)

**İniş Durumu:** `0` = Uygun Değil | `1` = Uygun  
**Hareket Durumu:** `-1` (N/A)

- 4,5 metre çaplı dairesel levha
- Alan **kısmen** görünse de **tespit** edilmelidir
- Iniş `uygun (1)` olabilmesi için alanın **tamamı** kare içinde olmalıdır
- Alan üzerinde herhangi bir nesne varsa (tespit edilsin ya da edilmesin) → `uygun değil (0)`
- Perspektif nedeniyle alana yakın görünen cisimler de → `uygun değil (0)`

---

### UAİ — Uçan Ambulans İniş (ID: 3)

**İniş Durumu:** `0` = Uygun Değil | `1` = Uygun  
**Hareket Durumu:** `-1` (N/A)

- UAP ile aynı kurallar geçerlidir
- Kırmızı renkli 4,5 metre çaplı dairesel levha

---

## İniş Durumu Tablosu

| İniş Durum ID | İniş Durumu    |
|---------------|----------------|
| 0             | Uygun Değil    |
| 1             | Uygun          |
| -1            | İniş Alanı Değil |

---

## Hareket Durumu Değerleri

| ID | Durum       |
|----|-------------|
| 0  | Hareketsiz  |
| 1  | Hareketli   |
| -1 | Geçersiz    |

## Genel Kurallar

1. Görüntü karesinin tamamındaki tüm nesneler tespit edilmelidir.
2. Kısmen görünen nesneler de dahil edilmelidir.
3. Başka nesnenin arkasında kısmen görünen nesneler de tespit edilmelidir.
4. RGB ve termal kamera görüntüleri desteklenmelidir.
5. Gündüz/gece, yağmur/kar, şehir/orman/deniz koşulları için hazırlıklı olunmalıdır.
