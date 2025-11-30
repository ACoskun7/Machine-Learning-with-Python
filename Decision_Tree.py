# -*- coding: utf-8 -*-  # Dosyada Türkçe karakter kullanabilmek için kodlama tanımı

# Iris veri setini yüklemek için
from sklearn.datasets import load_iris

# Veriyi eğitim ve test olarak bölebilmek için
from sklearn.model_selection import train_test_split

# Karar ağacı (Decision Tree) modeli ve görselleştirme fonksiyonu için
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Doğruluk (accuracy) ve karmaşıklık matrisi (confusion matrix) hesaplamak için
from sklearn.metrics import accuracy_score, confusion_matrix

# Grafik çizimleri için
import matplotlib.pyplot as plt


# Iris veri setini belleğe yükler
iris = load_iris()

# Özellikleri (features) X değişkenine atar (sepal/petal uzunluk-genişlik vb.)
X = iris.data  # features
# Sınıf etiketlerini (setosa, versicolor, virginica) y değişkenine atar
y = iris.target  # target 

# Veriyi %80 eğitim, %20 test olacak şekilde böler, random_state sonuçların tekrarlanabilir olmasını sağlar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gini kriterli, maksimum derinliği 5 olan bir Decision Tree sınıflandırıcı tanımlar
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)

# Karar ağacı modelini eğitim verisi ile eğitir (fit eder)
tree_clf.fit(X_train, y_train)

# Eğitilen model ile test verisi üzerinde tahmin yapar
y_pred = tree_clf.predict(X_test)

# Test etiketleri ile tahminleri karşılaştırarak doğruluk (accuracy) değeri hesaplar
accurarcy = accuracy_score(y_test, y_pred)

# Doğruluk skorunu ekrana yazdırır
print("iris veri seti ile eğitilen DT modeli doğruluğu: ", accurarcy)

# Gerçek ve tahmin etiketlerine göre confusion matrix hesaplar
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix'i ekrana yazdırır
print("conf matrix:")
print(conf_matrix)

# Yeni bir figür oluşturur
plt.figure()

# Karar ağacını görselleştirir, düğümleri renkli, özellik ve sınıf isimleri ile birlikte çizer
plot_tree(
    tree_clf,
    filled=True,  # sınıflara göre düğümleri renklendir
    feature_names=iris.feature_names,  # dallarda kullanılacak özellik adları
    class_names=list(iris.target_names)  # yapraklarda gösterilecek sınıf adları
)

# Grafiği ekranda gösterir
plt.show()

# Karar ağacının kullandığı her özelliğin önem skorlarını alır
feature_importances = tree_clf.feature_importances_

# Özellik adlarını kısaca almak için
feature_names = iris.feature_names

# (önem skoru, özellik adı) çiftlerini birleştirip önem skoruna göre azalan sıralar
feature_importances_sorted = sorted(
    zip(feature_importances, feature_names),
    reverse=True
)

# Her bir özelliğin önemini sırayla ekrana yazdırır
for importance, feature_name in feature_importances_sorted:
    print(f"feature name: {feature_name} importance: {importance}")
