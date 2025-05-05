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
Beberapa teknik data preparation yang diterapkan pada dataset ini:
1. Penghapusan Kolom yang Tidak Dibutuhkan :
   Menghapus kolom 'A_id' karena hanya berfungsi sebagai identifier dan tidak memberikan informasi yang berguna untuk prediksi kualitas apel.
2. Encoding Variabel Target : 
   Mengubah variabel target 'Quality' dari kategorikal menjadi numerik (0 untuk 'bad' dan 1 untuk 'good').
3. Pembagian Dataset :
   Dataset dibagi menjadi dua bagian: training set (80%) dan testing set (20%).
4. Standarisasi Data :
   Menggunakan StandardScaler untuk menormalisasi fitur numerik agar memiliki rata-rata 0 dan standar deviasi 1.

## Modeling
Pada proyek ini, algoritma Random Forest Classifier digunakan untuk memprediksi kualitas apel. Berikut adalah penjelasan tahapan dan parameter yang digunakan dalam proses pemodelan:

**Cara Kerja Random Forest Classifier:**
Random Forest adalah algoritma ensemble yang bekerja dengan cara:
1. Membuat Multiple Decision Trees : Algoritma ini membangun banyak pohon keputusan (decision trees) secara paralel.
2. Random Sampling : Setiap pohon dilatih menggunakan sampel acak dari data training dengan penggantian (bootstrap sampling).
3. Random Feature Selection : Pada setiap node pohon, hanya subset acak dari fitur yang dipertimbangkan untuk pemisahan.
4. Voting Majority : Untuk klasifikasi, hasil akhir ditentukan berdasarkan voting mayoritas dari semua pohon.

**Parameter yang Digunakan:**
- n_estimators: Jumlah pohon dalam Random Forest.
- max_depth: Maksimum kedalaman pohon.
- min_samples_split: Jumlah sampel minimum yang dibutuhkan untuk membagi node.
- min_samples_leaf: Jumlah sampel minimum yang diperlukan pada node terminal.

**Proses Hyperparameter Tuning:**
GridSearchCV digunakan untuk mencari kombinasi hyperparameter terbaik. Proses ini mencoba semua kombinasi hyperparameter yang ditentukan dalam grid dan memilih kombinasi yang memberikan performa terbaik berdasarkan metrik evaluasi yang ditentukan (misalnya, akurasi).
1. Definisikan Grid Parameter:
   Grid parameter adalah kombinasi hyperparameter yang akan dicoba oleh GridSearchCV. Berikut merupakan parameter yang akan dicoba:
   param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
  }
2. Cross-Validation:
   Cross-validation digunakan untuk mengevaluasi performa model dengan membagi data menjadi beberapa subset (fold). Menggunakan 5-fold cross-validation untuk mengevaluasi setiap kombinasi parameter.
3. Model Terbaik:
   Memilih kombinasi parameter yang menghasilkan akurasi tertinggi.

Hasil hyperparameter tuning menunjukkan parameter terbaik untuk Random Forest adalah:
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 1

## Evaluation
Berikut adalah hasil evaluasi model Random Forest yang terpilih:

- Accuracy
  - Definisi: Persentase prediksi yang benar dari total prediksi
  - Formula: (TP + TN) / (TP + TN + FP + FN)
  - Nilai: 90%
  Artinya, model berhasil memprediksi 90% dari total sampel dengan benar.

- Precision:
  - Formula: TP / (TP + FP)
  - Kelas 'bad': 90%
  - Kelas 'good': 89%
  Artinya, dari semua apel yang diprediksi berkualitas buruk, 90% benar-benar berkualitas buruk. Dan dari semua apel yang diprediksi berkualitas baik, 89% benar-benar berkualitas baik.

- Recall:
  - Formula: TP / (TP + FN)
  - Kelas 'bad': 89%
  - Kelas 'good': 90%
  Artinya, dari semua apel yang sebenarnya berkualitas buruk, 89% berhasil diidentifikasi dengan benar. Dan dari semua apel yang sebenarnya berkualitas baik, 90% berhasil diidentifikasi dengan benar.

- F1-Score:
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Kelas 'bad': 90%
  - Kelas 'good': 90%
  F1-Score memberikan keseimbangan antara precision dan recall, dan nilai 90% untuk kedua kelas menunjukkan model memiliki keseimbangan yang baik.

Confusion matrix memberikan gambaran visual tentang performa model:
[[ 358   43 ]
 [  40  359 ]]
- 358 apel berkualitas buruk diprediksi dengan benar (True Negative)
- 359 apel berkualitas baik diprediksi dengan benar (True Positive)
- 43 apel berkualitas buruk diprediksi salah sebagai baik (False Positive)
- 40 apel berkualitas baik diprediksi salah sebagai buruk (False Negative)

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
