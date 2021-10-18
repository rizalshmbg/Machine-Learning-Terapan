# Laporan Proyek Machine Learning - Rizal Sihombing

## Domain Proyek
Tenggelamnya Titanic adalah sebuah bencana kapal pesiar yang terkenal pada pelayaran perdananya. Kapal RMS Titanic yang digadang-gadang "tidak dapat tenggelam", akhirnya tenggelam di bagian utara samudera Atlantik pada tanggal 15 April 1912. Dan menewaskan 1502 dari 2224 penumpang dan awak kapal. Meskipun ada beberapa elemen keberuntungan yang terlibat untuk selamat, tampaknya beberapa kelompok orang lebih mungkin untuk selamat daripada yang lain. Analisis data tentang "orang seperti apa yang lebih mungkin untuk selamat?"  menggunakan data penumpang (yaitu nama, usia, jenis kelamin, kelas tiket, dll). Pendekatan yang dilakukan adalah dengan memanfaatkan kumpulan data umum yang tersedia dari situs web yang dikenal seperti [Kaggle](https://www.kaggle.com/c/titanic/data).

Referensi :
* http://csis.pace.edu/~ctappert/srd2014/d3.pdf
* https://www.researchgate.net/profile/Neytullah-Acun/publication/324909545_A_Comparative_Study_on_Machine_Learning_Techniques_Using_Titanic_Dataset/links/607533bc299bf1f56d51db20/A-Comparative-Study-on-Machine-Learning-Techniques-Using-Titanic-Dataset.pdf


## Business Understanding
Kumpulan data ini merekam berbagai fitur penumpang di Titanic, termasuk siapa yang selamat dan siapa yang tidak. Disadari bahwa beberapa fitur yang hilang dan tidak berkorelasi mengurangi kinerja prediksi. Untuk analisis data rinci, efek dari fitur telah diselidiki. Jadi beberapa fitur baru ditambahkan ke dataset dan beberapa fitur yang ada dihapus dari kumpulan data.

### Problem Statements (Pernyataan Masalah)
Apakah penumpang kapal Titanic selamat atau tidak? \
Mengetahui penumpang seperti apa yang lebih mungkin untuk selamat?

### Goals (Tujuan)
* Mengetahui fitur yang paling berpengaruh terhadap keselamatan penumpang.
* Untuk mendapatkan hasil yang dapat mendekati prediksi dari data mentah, dengan menggunakan pembelajaran mesin dan metode rekayasa fitur.

### Solution statements (Pernyataan Solusi)
Mengetahui fitur apa saja yang berpengaruh terhadap keselamatan penumpang Titanic dan dapat memprediksi penumpang yang selamat dan tidak selamat. Maka, metodologi pada proyek ini adalah membangun model regresi dengan fitur penumpang sebagai target. Dan, memprediksi penumpang yang dapat selamat dan tidak selamat dengan klasifikasi. Menggunakan :
1. KNN (K-Nearest Neighbor) \
Mengklasifikasikan objek baru berdasarkan atribut dan sampel-sampel dari pelatihan data.
2. Decision Tree \
Prediksi menggunakan struktur pohon atau struktur berhirarki. Lalu mengeliminasi perhitungan atau data-data yang tidak diperlukan. Sebab, sampel yang ada biasanya hanya diuji berdasarkan kriteria atau kelas tertentu saja.
3. Random Forest \
Merupakan salah satu metode dalam Decision Tree, dan kombinasi dari masing-masing tree yang diperlukan kemudian dikombinasikan ke dalam satu model. Random Forest bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing-masing decision tree memiliki kedalaman yang maksimal.
4. Super Vector Machine (Classifier) \
Algoritma yang bertujuan untuk memaksimalkan margin antara pola pelatihan dan batas kepututsan, dengan sebuah bidang yang mampu memisahkan dua buah kelas.

## Data Understanding
Data yang digunakan pada proyek kali ini adalah Titanic dataset, yang dapat diunduh dari [Kaggle](https://www.kaggle.com/c/titanic/data). \
train.csv berjumlah 891 kolom, sedangkan test.csv 418 kolom

### Variabel atau fitur pada Titanic dataset adalah sebagai berikut :
* PassengerId : Id penumpang
* Pclass : kelas tiket. 1 = 1st, 2 = 2nd, 3 = 3rd
* Survived : penumpang yang selamat,0 = Tidak, 1 = Ya
* Name : nama penumpang
* Sex : jenis kelamin penumpang. 0 = laki-laki, 1 = perempuan
* Age : umur penumpang
* SibSp : Sibling & Spouse yang artinya jumlah saudara kandung atau pasangan yang dibawa oleh penumpang
* Parch : jumlah keluarga atau anak yang dibawa oleh penumpang
* Ticket : id atau nomor tiket penumpang
* Fare : tarif tiket. 0 = kurang dari 17 Poundsterling (UK), 1 = lebih dari 17 Poundsterling (UK), 2 = lebih dari 30 Poundsterling (UK), 3 = lebih dari 100 Poundsterling (UK)
* Cabin : kategori kabin. A = 0.0, B = 0.4, C = 0.8, D = 1.2, E = 1.6, F = 2.0, G = 2.4, T = 2.8
* Embarked : dari pelabuhan mana penumpang naik. C = Cherbourg, Q = Queenstown, S = Southampton
* Title : pengelompokkan penumpang contohnya apakah penumpang tersebut sudah menikah atau belum. 0 = Mr, 1 = Miss, 2 = Mrs, 3 = Lainnya
* FamilySize : jumlah dari SibSp ditambah dengan Parch. 1= 0, 2= 0.4, 3= 0.8, 4= 1.2, 5= 1.6, 6= 2, 7= 2.4, 8= 2.8, 9= 3.2, 10= 3.6, 11= 4


## Data Preparation
Pada tahap ini saya menerapkan proses :
* Data Cleaning, karena ada beberapa kolom yang berisi nilai NaN
* Data Integration, saya menggabungkan kolom SibSp dengan Parch menjadi NumOfFamily karena dua kolom ini memiliki pemahaman yang sama
* Data Transformation, saya menggunakan teknik ini pada kolom Name menjadi Title untuk mempermudah pembacaan fitur dan memperoleh data yang lebih berkualitas


## Modeling
Pada tahap ini, saya mengembangkan model machine learning dengan empat algoritma. Kemudian mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik.
* K-Nearest Neighbor
* Decision Tree
* Random Forest
* Super Vector Machine (Classifier)

Fungsi yang digunakan adalah cross-val-score, yaitu salah satu teknik yang digunakan untuk menguji keefektifan model. Model dengan peforma terbaik adalah Super Vector Machine (Classifier), yang memiliki nilai akurasi sebesar 83.5.


## Evaluation
Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Di proyek ini saya menggunakan **precision, recall, accuracy, dam F1 score**
* Precission : mewakili kemampuan model untuk memprediksi positif dengan benar dari semua prediksi positif yang dibuatnya. Skor presisi adalah ukuran yang berguna dari keberhasilan prediksi ketika kelas sangat tidak seimbang. Secara matematis, ini mewakili rasio positif benar dengan jumlah positif benar dan positif palsu.
  * Formulanya adalah TP/(FP + TP), jadi 256/(58+256) = 0.815
* Cara mengimplementasikannya adalah dengan method precision_score dari sklearn.metrics
* Recall : kemampuan model untuk memprediksi dengan benar hal-hal positif dari hal-hal positif yang sebenarnya. Tidak seperti precision sisi yang mengukur berapa banyak prediksi yang dibuat oleh model yang benar-benar positif dari semua prediksi positif yang dibuat.
  * Formulanya adalah TP/(FN + TP), jadi 256/(86+256) = 0.749
* Cara mengimplementasikannya adalah dengan method recall_score dari sklearn.metrics
* Accuracy : kinerja model yang didefinisikan sebagai rasio positif dan negatif sejati untuk semua pengamatan positif dan negatif.
  * Formulanya adalah (TP + TN)/(TP + FN + TN + FP), jadi (256+491)/(256+86+491+58) = 0.838
* Cara mengimplementasikannya adalah dengan method accuracy_score dari sklearn.metrics
* F1 Score : metrik kinerja model yang memberikan bobot yang sama untuk Precision dan Recall untuk mengukur kinerjanya dalam hal akurasi, menjadikannya alternatif untuk metrik akurasi
  * Formulanya adalah 2 * Precision * Recall / Precision + Recall = 2 * 0.815 * 0.749 / 0.815 + 0.749 = 0.780
* Cara mengimplementasikannya adalah dengan method f1_score dari sklearn.metrics
