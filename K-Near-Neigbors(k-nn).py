# sklearn - ml library 

# Meme kanseri veri setini yüklemek için
from sklearn.datasets import load_breast_cancer
# K-En Yakın Komşu sınıflandırıcı (KNN Classifier) için
from sklearn.neighbors import KNeighborsClassifier
# Doğruluk (accuracy) ve confusion matrix hesaplamak için
from sklearn.metrics import accuracy_score, confusion_matrix
# Veriyi eğitim ve test setine bölmek için
from sklearn.model_selection import train_test_split
# Özellikleri ölçeklendirmek (standardizasyon) için
from sklearn.preprocessing import StandardScaler
# Veri çerçevesi (tablo) yapısı için
import pandas as pd 

# Grafik çizimleri için
import matplotlib.pyplot as plt

# 1- Veri setini inceleme (investigation the data set) 
cancer = load_breast_cancer()  # Breast cancer veri setini sklearn içinden yükler

# Veri setini pandas DataFrame formatına çevirir, sütun isimlerini feature_names olarak ayarlar
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
# Hedef (0 = malignant, 1 = benign) sütununu ekler
df["target"] = cancer.target


# 2- Makine öğrenmesi modeli seçimi  -- KNN Classification seçildi
# (Bu kısımda sadece yorumla belirtilmiş, henüz model oluşturulmadı)


# 3- Model eğitimi (training)

X = cancer.data   # features: giriş değişkenleri (ölçülen özellikler)
y = cancer.target # target: sınıf etiketleri (kanser tipi)

# Veriyi eğitim (%70) ve test (%30) olarak böler, random_state sonuçların sabit kalmasını sağlar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizasyon (özellikleri aynı ölçeğe çekmek için)
scaler = StandardScaler()               # StandardScaler nesnesi oluşturur
X_train = scaler.fit_transform(X_train) # Eğitim verisine göre ortalama-standart sapma hesaplar ve dönüştürür
X_test = scaler.transform(X_test)       # Test verisini, eğitimde hesaplanan değerlere göre dönüştürür


# KNN modelini oluştur ve eğit  !!

knn = KNeighborsClassifier(n_neighbors=3)  # k=3 komşulu KNN sınıflandırıcı modeli oluşturur

# fit fonksiyonu, eğitim verisi (X_train, y_train) ile modeli eğitir
knn.fit(X_train, y_train)  


# 4 - Sonuçların değerlendirilmesi (evaluation)

y_pred = knn.predict(X_test)      # Test seti üzerindeki sınıf tahminlerini üretir

accurarcy = accuracy_score(y_test, y_pred)  # Tahmin ile gerçek etiketleri karşılaştırıp doğruluk oranını hesaplar
print("Accurarcy: ", accurarcy)             # Doğruluk oranını ekrana yazdırır

conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix hesaplar
print("Confusion matrix: ")
print(conf_matrix)                              # Confusion matrix'i ekrana basar


# 5- Hyperparameter tuning (K değerini test etme)
"""
KNN : hyperparameter = K
k:1,2,3,...n
Accuracy : %a,%b ,%c, ...

Burada farklı k değerleri için doğruluk oranı karşılaştırılacak
"""

accurarcy_values = []  # Her k için hesaplanan doğruluk değerlerini tutmak için liste
k_values = []          # Denenen k değerlerini tutmak için liste

# k değerlerini 1'den 20'ye kadar dene
for k in range(1, 21):
    
    knn = KNeighborsClassifier(n_neighbors=k)  # Her döngüde yeni bir KNN modeli oluşturur
    knn.fit(X_train, y_train)                 # Modeli eğitim verisi ile tekrar eğitir
    y_pred = knn.predict(X_test)              # Test verisi üzerinde tahmin yapar
    accurarcy = accuracy_score(y_test, y_pred) # Doğruluk oranını hesaplar
    accurarcy_values.append(accurarcy)        # Doğruluk oranını listeye ekler
    k_values.append(k)                        # K değerini listeye ekler


# k'ya göre doğruluk grafiğini çizer
plt.figure()  # Yeni bir figür açar
plt.plot(k_values, accurarcy_values, marker="o", linestyle="-")  # k vs accuracy çizdirir
plt.title("Accurarcy values according to K values")              # Grafiğin başlığını ayarlar
plt.xlabel("k_values")                                          # x-ekseni etiketini ayarlar
plt.ylabel("Accurarcy")                                         # y-ekseni etiketini ayarlar
plt.xticks(k_values)                                            # x-ekseni tick'lerini her k değeri için ayarlar
plt.grid(True)                                                  # Arka plan ızgarasını açar


# %% KNN REGRESSOR ÖRNEĞİ

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor  # KNN ile regresyon yapmak için

# 0–5 aralığında 40 tane rastgele X değeri üretir ve sıralar, bunları feature (girdi) olarak kullanır
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # features

# Hedef (y) değerlerini sinüs fonksiyonuna göre üretir ve tek boyuta indirger
y = np.sin(X).ravel()  # target
# plt.scatter(X, y)  # Veriyi görmek için noktaları çizebilirdiniz (yorum satırına alınmış)

# Her 5. örnek için y değerine rastgele gürültü ekleyerek veriyi biraz bozar (realistik hale getirir)
y[::5] += 1 * (0.5 - np.random.rand(8))
# plt.scatter(X, y)  # Gürültülü veriyi görmek için noktaları çizebilirdiniz (yorum satırına alınmış)

# 0–5 aralığında, 500 noktalı bir grid oluşturur; bunlar üzerinde tahmin yapılacak
T = np.linspace(0, 5, 500)[:, np.newaxis]

# KNN Regressor için iki farklı ağırlık (weights) stratejisini dener: "uniform" ve "distance"
for i, weigth in enumerate(["uniform", "distance"]):
    # n_neighbors=5 ve ilgili weights ile KNN Regressor modeli oluşturur
    knn = KNeighborsRegressor(n_neighbors=5, weights=weigth)
    # Modeli (X, y) verisiyle eğitir ve T noktaları üzerinde tahmin yapar
    y_pred = knn.fit(X, y).predict(T)
    
    # 2 satır 1 sütunluk subplot yapısında i. grafiği seçer
    plt.subplot(2, 1, i + 1)
    # Orijinal (gürültülü) veriyi yeşil noktalarla çizer
    plt.scatter(X, y, color="green", label="data")
    # Tahmin edilen eğriyi mavi çizgiyle çizer
    plt.plot(T, y_pred, color="blue", label="prediction")
    # Grafik eksenlerini veriye sıkı oturtur
    plt.axis("tight")
    # Legend (açıklama kutusu) ekler
    plt.legend()
    # Grafiğin başlığında hangi weight stratejisinin kullanıldığını gösterir
    plt.title("KNN Regressor Weights = {}".format(weigth))
    
# Alt alta olan iki subplot arasındaki boşlukları daha düzenli hale getirir
plt.tight_layout()
# Tüm grafikleri ekranda gösterir
plt.show()
