"""
Sınıf Tanımları ve Yarışma Kuralları
Havacılıkta Yapay Zekâ Yarışması — Nesne Tespiti
"""

# Sınıf sözlüğü
CLASSES = {
    "tasit": {
        "id": 0,
        "tr": "Taşıt",
        "motion_values": [0, 1],
        "landing_value": -1,
        "description": (
            "Motorlu karayolu taşıtları (otomobil, motosiklet, otobüs, kamyon, traktör, ATV), "
            "raylı taşıtlar (tren, lokomotif, vagon, tramvay, monoray, füniküler), "
            "tüm deniz taşıtları."
        ),
        "notes": [
            "Tren lokomotifi ve vagonları AYRI AYRI taşıt olarak etiketlenir.",
            "Bisiklet ve motosiklet sürücüsü 'insan' olarak etiketlenmez; taşıt+sürücü = taşıt.",
            "Scooter: sürücüsüz → taşıt, sürücülü → insan.",
        ],
    },
    "insan": {
        "id": 1,
        "tr": "İnsan",
        "motion_value": -1,
        "landing_value": -1,
        "description": "Ayakta duran ya da oturan fark etmeksizin tüm insanlar.",
        "notes": [
            "Bisiklet/motosiklet sürücüsü insan değil, taşıt olarak etiketlenir.",
            "Kısmen görünen insanlar da tespit edilmelidir.",
        ],
    },
    "uap": {
        "id": 2,
        "tr": "Uçan Araba Park",
        "motion_value": -1,
        "landing_values": [0, 1],
        "description": "Uçan arabaların park edebileceği alanlar.",
    },
    "uai": {
        "id": 3,
        "tr": "Uçan Ambulans İniş",
        "motion_value": -1,
        "landing_values": [0, 1],
        "description": "Ambulans uçaklarının inebileceği iniş alanları.",
    },
}

MOTION_LABELS = {0: "Hareketsiz", 1: "Hareketli", -1: "N/A"}
LANDING_LABELS = {0: "İnilemez", 1: "İnilebilir", -1: "N/A"}
