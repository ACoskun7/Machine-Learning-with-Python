# -*- coding: utf-8 -*-
# Bu kod, Iris veri seti üzerinde Decision Tree (karar ağacı) sınıflandırıcısının
# farklı özellik çiftleri (feature pairs) için nasıl karar verdiğini görselleştirir.
# Her özellik çifti için ayrı bir karar sınırı (decision boundary) çizilir ve
# gerçek veri noktaları bu sınırlar üzerinde gösterilerek:
# - Özelliklerin sınıfları ne kadar iyi ayırdığı
# - Karar ağacının uzayı nasıl parçalara böldüğü
# gözlemlenebilir. Eğitim amaçlı, model davranışını anlamaya yönelik bir koddur.
# Iris veri setindeki her özellik çifti için, bir Decision Tree modelinin 2 boyutlu uzayı nasıl parçalara ayırdığını (karar sınırlarını) ve gerçek veri noktalarının bu bölgelerde nasıl dağıldığını görselleştirmek.

# Iris veri setini yüklemek için
from sklearn.datasets import load_iris  # hazır iris veri seti

# Karar ağacı (Decision Tree) modeli için
from sklearn.tree import DecisionTreeClassifier  # sınıflandırma modeli

# Grafik çizimleri için
import matplotlib.pyplot as plt  # grafik çizmek için
from sklearn.inspection import DecisionBoundaryDisplay  # karar sınırı çizimi
import numpy as np  # sayısal işlemler için


# veri setini inceleme
iris = load_iris()  # iris veri setini belleğe yükler

n_classes = len(iris.target_names)  # sınıf sayısını (3 tür iris) hesaplar
plot_colors = "ryb"  # sınıflar için renkler: r=red, y=yellow, b=blue

# Tüm özellik çiftleri için döngü: [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # X: seçilen iki özelliğin (pair) sütunları
    X = iris.data[:, pair]  # sadece 2 feature alıyoruz (2D çizim için)
    # y: hedef sınıf etiketleri (0,1,2)
    y = iris.target

    # Decision Tree modelini bu iki özellik ile eğit
    clf = DecisionTreeClassifier().fit(X, y)  # model öğrenme (fit)

    # 2 satır 3 sütunluk subplot düzeninde ilgili ekseni seç
    ax = plt.subplot(2, 3, pairidx + 1)  # her özellik çifti için bir grafik
    # Grafikler arasında boşlukları ayarla (daha okunaklı görüntü)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # Karar sınırı (decision boundary) alanını çiz
    DecisionBoundaryDisplay.from_estimator(
        clf,                   # eğitilmiş karar ağacı modeli
        X,                     # eğitim verisi (2 boyut)
        cmap=plt.cm.RdYlBu,    # renk haritası (kırmızı-sarı-mavi)
        response_method="predict",  # her noktayı hangi sınıfa atadığını kullan
        ax=ax,                 # bu karar sınırını hangi subplot’a çizeceğini belirt
        xlabel=iris.feature_names[pair[0]],  # x ekseni etiketi (özellik adı)
        ylabel=iris.feature_names[pair[1]],  # y ekseni etiketi (özellik adı)
    )

    # Her sınıfın gerçek veri noktalarını aynı grafiğe ekle
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)  # sınıf i’ye ait indeksleri bul
        plt.scatter(
            X[idx, 0],              # seçilen çiftin 1. özelliği (x ekseni)
            X[idx, 1],              # seçilen çiftin 2. özelliği (y ekseni)
            c=color,                # sınıfa özel renk
            label=iris.target_names[i],  # lejand etiketi (setosa / versicolor / virginica)
            cmap=plt.cm.RdYlBu,     # renk haritası (zorunlu değil ama uyumlu)
            edgecolors="black",     # noktaların kenar rengi (daha belirgin görünür)
        )

# Tüm grafikler için ortak legend (sınıf isimleri)
plt.legend()  # grafiklerin altına sınıf isimlerini göster
