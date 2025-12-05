# -*- coding: utf-8 -*-
"""
Bu kod, sklearn load_diabetes veri seti üzerinde bir Decision Tree Regressor 
modeli kurarak, modeli eğitme–tahmin etme–hata hesaplama (MSE, RMSE) süreçlerini göstermektedir. 
Temel amaç: regresyon modelinin performansını değerlendirmek."""

from sklearn.datasets  import load_diabetes
# sklearn içinde hazır gelen diabetes veri setini yüklemek için import edilir.

from sklearn.model_selection import train_test_split
# Veri setini eğitim ve test olarak ayırmak için kullanılır.

from sklearn.tree import DecisionTreeRegressor
# Karar ağacı tabanlı regresyon modeli sınıfı.

from sklearn.metrics import mean_squared_error
# Modelin tahmin doğruluğunu ölçmek için MSE metriğini sağlar.

import numpy as np
# Sayısal işlemler (özellikle karekök alma için) kullanılır.


# --- Veri setinin yüklenmesi ---
diabetes = load_diabetes()
# Diabetes veri seti yüklenir (X: özellikler, y: hedef değişken).

X = diabetes.data
# Özellik matrisi (10 adet klinik değişken yer alır).

y = diabetes.target
# Hedef değişken: diyabet progresyon değerleri.


# --- Veri setinin eğitim/test olarak ayrılması ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2 → verinin %20’si test olarak ayrılır.
# random_state=42 → tekrar çalıştırıldığında aynı sonuçların alınmasını sağlar.


# --- Karar ağacı regresyon modeli oluşturma ---
tree_reg = DecisionTreeRegressor(random_state=42)
# Karar ağacı modeli oluşturulur. random_state deterministik yapı sağlar.

tree_reg.fit(X_train, y_train)
# Model, eğitim verisiyle öğrenme sürecini gerçekleştirir.


# --- Test verisi üzerinde tahmin ---
y_pred = tree_reg.predict(X_test)
# Model test verisi için tahmin üretir.


# --- MSE hesaplama ---
mse = mean_squared_error(y_test, y_pred)
print("mse: ", mse)
# Tahmin ile gerçek değerler arasındaki ortalama kare hata hesaplanır.


# --- RMSE hesaplama ---
rmse = np.sqrt(mse)
print("rmse: ", rmse)
# RMSE: Hatanın karekökü. Yorumlaması daha kolay bir regresyon metriğidir.
