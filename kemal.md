# Laporan Proyek Machine Learning – Kemal Aziz

## 1. Domain Proyek

### 1.1. Latar Belakang
Industri pertanian, khususnya produksi buah apel, merupakan sektor penting dalam ekonomi global. Kualitas apel sangat menentukan nilai jualnya di pasar dan kepuasan konsumen. Penentuan kualitas apel secara tradisional dilakukan melalui inspeksi visual oleh manusia, yang bisa subjektif dan memakan waktu. Dengan kemajuan teknologi sensor dan machine learning, kini dimungkinkan untuk mengotomatisasi proses klasifikasi kualitas apel berdasarkan berbagai parameter fisik dan kimia.

### 1.2. Mengapa Masalah Ini Penting?

Klasifikasi kualitas apel secara otomatis memiliki beberapa manfaat penting:

1. Meningkatkan efisiensi dalam rantai pasok industri buah
2. Mengurangi subjektivitas dalam penilaian kualitas
3. Membantu petani dan distributor dalam menentukan harga yang sesuai
4. Memastikan konsumen mendapatkan produk dengan kualitas yang konsisten

Dengan menggunakan machine learning untuk memprediksi kualitas apel, kita dapat mengembangkan sistem yang lebih cepat, konsisten, dan akurat dalam menilai kualitas apel dibandingkan metode manual.

Sumber Refrensi:\
[Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) 
[Research Paper: Machine Learning For Fruit Quality assessment](https://www.mdpi.com/2073-4395/10/4/556)


## 2. Business Understanding
### 2.1 Problem Statements
Bagaimana cara mengembangkan model machine learning yang dapat memprediksi kualitas apel (baik/buruk) berdasarkan karakteristik fisik dan kimia yang terukur?

### 2.2. Goals
Membangun model klasifikasi dengan akurasi minimal 80% untuk mengklasifikasikan kualitas apel berdasarkan parameter yang terukur.

### 2.3 Solution statements
- Mengembangkan model Random Forest Classifier sebagai algoritma utama karena kemampuannya menangani data numerik dengan baik dan ketahanannya terhadap overfitting.
- Melakukan hyperparameter tuning pada model untuk meningkatkan performa.
- Memilih model terbaik berdasarkan metrik evaluasi akurasi, precision, recall, dan F1-score.

## 3. Data Understanding
[Dataset dari UCI Machine LEarning](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)\
Dataset yang digunakan berisi 4000 sampel apel dengan berbagai karakteristik fisik dan kimia.

Fitur/Variabel Utama:
* A_id: ID unik untuk setiap sampel apel
* Size: Ukuran apel (nilai numerik)
* Weight: Berat apel (nilai numerik)
* Sweetness: Tingkat kemanisan (nilai numerik)
* Crunchiness: Tingkat kerenyahan (nilai numerik)
* Juiciness: Tingkat keempukan (nilai numerik)
* Ripeness: Tingkat kematangan (nilai numerik)
* Acidity: Tingkat keasaman (nilai numerik)
* Quality: Kualitas apel (good/bad) - target variabel

Exploratory Data Analysis (EDA):
Berikut adalah beberapa temuan dari analisis eksplorasi data:

1. Distribusi Target Variable :
   Dari 4000 sampel, sekitar 50.1% apel berkualitas baik (good) dan 49.9% berkualitas buruk (bad), menunjukkan dataset yang cukup seimbang.
2. Korelasi antar Fitur :
   - Ripeness memiliki korelasi positif yang kuat dengan kualitas apel
   - Size juga menunjukkan korelasi positif dengan kualitas
   - Juiciness memiliki korelasi negatif dengan kualitas
3. Distribusi Fitur berdasarkan Kualitas :
   - Apel berkualitas baik cenderung memiliki tingkat kematangan (Ripeness) yang lebih tinggi
   - Apel berkualitas baik umumnya memiliki tingkat ukuran (Size) yang lebih besar
   - Apel berkualitas buruk cenderung memiliki tingkat keempukan (Juiciness) yang lebih tinggi

## Data Preparation
Pada dataset yang digunakan pada proyek ini, data sudah clean tanpa ada missing values.
Data preparation yang kita lakukan ini hanya mengubah variabel target 'Quality' dari kategorikal (good/bad) menjadi numerik (1/0) serta membagi datatest sebanyak 1:4 yatu 80% data train, 20% data test. Pembagian proporsi ini umum dilakukan untuk mengukur kinerja data baru yang masuk. 
Selain itu, tahap persiapan data yang kita lakukan adalah standarisasi data dengan menggunakan metode StandarScaler. Prinsipnya adalah standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Metode ini membantu mengurangi variasi dalam data dan meningkatkan performa model.

## Modeling
Model yang dirancang untuk menyelesaikan masalah ini dengan algoritma Random Forest. Model ini dipilih karena menggunakan multiple decision trees untuk menghasilkan prediksi yang lebih akurat dan stabil.
Dengan Random Forest, maka akan ada klasifikasi hyperlane yang memisahkan antara kategori anggur Good / Bad.\
Pada uji coba kali ini, model dioptimasi juga dengan metode [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Tujuan optimasi ini untuk menentukan Hyperparameter pada model Random Forest. GridSearchCV mengambil parameter yang sudah didefinisikan untuk dicoba pada model untuk tahap data training. Dengan hyperparameter yang optimal maka hasil akurasi akan meningkat juga. Uji coba kali ini, parameter yg akan ditentukan hyperparameternya adalah n_estimators, max_depth, min_samples_split, dan min_sample_leaf (merujuk pada parameter yang sesuai basic dokumentasi [berikut](https://scikit-learn.org/stable/modules/grid_search.html)

## Evaluation
Kelebihan Random Forest:
- Mampu menangani data numerik dengan baik
- Tahan terhadap overfitting
- Dapat menangani dataset dengan banyak fitur
- Memberikan informasi tentang feature importance

Kekurangan Random Forest:
- Komputasi yang lebih berat dibandingkan model yang lebih sederhana
- Kurang interpretable dibandingkan decision tree tunggal
- Memerlukan tuning hyperparameter untuk performa optimal

Hasil hyperparameter tuning menunjukkan parameter terbaik untuk Random Forest adalah:
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 1

Berikut adalah hasil evaluasi model Random Forest yang terpilih:
- Accuracy: 90% Artinya, 90% dari seluruh prediksi model adalah benar.
- Precision:
  - Kelas 'bad': 90%
  - Kelas 'good': 89%
  Artinya, dari semua apel yang diprediksi berkualitas buruk, 90% benar-benar berkualitas buruk. Dan dari semua apel yang diprediksi berkualitas baik, 89% benar-benar berkualitas baik.

- Recall:
  - Kelas 'bad': 89%
  - Kelas 'good': 90%
  Artinya, dari semua apel yang sebenarnya berkualitas buruk, 89% berhasil diidentifikasi dengan benar. Dan dari semua apel yang sebenarnya berkualitas baik, 90% berhasil diidentifikasi dengan benar.

- F1-Score:
  - Kelas 'bad': 90%
  - Kelas 'good': 90%
  F1-Score memberikan keseimbangan antara precision dan recall.

Confusion matrix memberikan gambaran visual tentang performa model:
[[ 358   43 ]
 [  40  359 ]]
- 358 apel berkualitas buruk diprediksi dengan benar (True Negative)
- 359 apel berkualitas baik diprediksi dengan benar (True Positive)
- 43 apel berkualitas buruk diprediksi salah sebagai baik (False Positive)
- 48 apel berkualitas baik diprediksi salah sebagai buruk (False Negative)

## Feature Importance
 Analisis feature importance dari model Random Forest menunjukkan kontribusi relatif setiap fitur:

1. Ripeness (16.7%): Fitur paling penting dalam menentukan kualitas apel
2. Size (16.2%): Fitur kedua terpenting
3. Juiciness (15.6%)
4. Sweetness (15.1%)
5. Acidity (13.8%)
6. Weight (11.8%)
7. Crunchiness (11.0%)

**---Ini adalah bagian akhir laporan---**
