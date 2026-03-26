# Görev 3: Görüntü Eşleme (Image Matching)

Bu görevde, daha önce tanımlanmamış "tanımsız nesneler" uçuş esnasında tespit edilir.

## Süreç
1. **Oturum Başlangıcı**: Sunucu bir dizi "referans nesne" görüntüsü paylaşır.
2. **Kare İşleme**: Her video karesinde bu referanslar aranır.
3. **Sonuç**: Bulunan nesnelerin klas adı (id) ve koordinatları (bbox) sunucuya gönderilir.

## Algoritma Detayları
Sistem aşağıdaki zorluklara dayanıklı olacak şekilde tasarlanmıştır:
- **Çoklu Ölçek (Multi-scale)**: Farklı irtifalardan çekim.
- **Çoklu Açı (Multi-angle)**: Diferansiyel rotasyonlar.
- **Çapraz-Modal**: RGB - Termal arası eşleme (CLAHE normalizasyonu ile).
- **Homografi**: Nesnenin perspektifsel bozulmalarına rağmen bulunması.

## Dosyalar
- `src/matching/matcher.py`: Ana eşleştirme mantığı.
- `src/matching/feature_extractor.py`: SIFT/ORB tabanlı özellik çıkarımı.
- `scripts/run_session.py`: Yarışma oturumu başlatıcı.
