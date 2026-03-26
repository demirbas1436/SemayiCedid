# Görev 2: Pozisyon Tespiti — Şartname Notları

## Amaç

Hava aracının konumlandırma sistemi (GPS vb.) güvenilmez hale geldiğinde,
**yalnızca kamera görüntüleri** ile x, y, z pozisyonu kestirmek.

## Oturum Akışı

| Aşama               | Kare Aralığı | Sağlık | Ne Yapılacak                           |
|----------------------|--------------|--------|----------------------------------------|
| İlk 1 dakika         | 0–449        | `1`    | Sunucu pozisyonu güvenilir, referans kaydet |
| Son 4 dakika          | 450–2249     | `0/1`  | Sağlık 0 ise kendin hesapla            |

## Sunucudan Alınan Bilgiler (Her Kare)

| Alan            | Açıklama                                      |
|-----------------|-----------------------------------------------|
| Video Karesi    | jpg/png görüntü                               |
| Pozisyon X      | Referans koordinatta X yer değiştirmesi (m)   |
| Pozisyon Y      | Referans koordinatta Y yer değiştirmesi (m)   |
| Pozisyon Z      | Referans koordinatta Z yer değiştirmesi (m)   |
| Sağlık          | 1=güvenilir, 0=güvenilmez                     |

## Sağlık Kuralları

- `Sağlık = 1`: Sunucu değerini aynen gönderebilirsin VEYA kendi kestirimin gönderebilirsin
- `Sağlık = 0`: Kendi kestirimin gönderMELİSİN → Visual Odometry devreye girer
- İlk 450 kare kesinlikle sağlıklıdır
- Son 1800 karede ne zaman sağlıksız olacağı belirsizdir

## Algoritma Çalışma Şartları

- Önceki kareye sonuç gönderilmeden yeni kare istenemez
- Her kare için 1 adet pozisyon sonucu gönderilmeli
- Toplu indirme mümkün değildir

## Hata Hesabı

Referans ile kestirim arasındaki X, Y, Z hataları ölçülür.
Düşük hata = yüksek puan.
