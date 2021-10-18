# Laporan Proyek Machine Learning - Muhammad Ikhlas Naufalsyah Ranau

## Domain Proyek

Akses untuk air minum yang aman sangat penting untuk kesehatan, hak asasi manusia dan komponen kebijakan yang efektif untuk perlindungan kesehatan. Hal ini penting sebagai masalah kesehatan dan pembangunan di tingkat nasional, regional dan lokal. Di beberapa daerah, telah terbukti bahwa investasi dalam penyediaan air dan sanitasi dapat menghasilkan keuntungan ekonomi bersih, karena pengurangan efek kesehatan yang merugikan dan biaya perawatan kesehatan lebih besar daripada biaya untuk melakukan intervensi.

referensi

- [Keuntungan Ekonomi terhadap Proyek Penyediaan Air Minum untuk Rumah Tangga di Negara Berkembang](https://www.adb.org/publications/economic-benefits-potable-water-supply-projects-households-developing-countries)
- [standar kualitas air minum WHO](https://www.who.int/publications/i/item/9789241549950)

## Business Understanding

Air minum yang aman sangat penting bagi kesehatan, tetapi masih banyak yang tidak mendapatkan air yang layak untuk diminum karena tidak mengetahui apakah air minum layak dikonsumsi atau tidak. berangkat dari masalah itu untuk memudahkan mengetahui antara air bisa diminum atau tidak maka dibuatlah model machine learning yang bisa membedakan antara air layak untuk diminum atau tidak.

### Problem Statements

Mengklasifikasikan kualitas air apakah layak minum atau tidak berdasarkan fitur yang disediakan.

### Goals

- Model dapat memprediksi apakah air minum layak untuk diminum atau tidak
- Model mempunyai akurasi yang lebih dari 0.85

### Solution statements

Solusi pada kasus kali ini adalah dengan menggunakan beberapa model machine learning untuk memprediksi apakah air minum dapat di minum atau tidak. Penggunaan beberapa model berfungsi untuk bisa membandingkan model mana yang lebih baik dalam menyelesaikan masalah klasifikasi air minum. model yang akan digunakan yaitu:

1. **Logistic Regression**\
   Logistic Regression adalah sebuah algoritma klasifikasi untuk mencari hubungan antara fitur (input) diskrit/kontinu dengan probabilitas hasil output diskrit tertentu.

2. **K-Nearest Neighbor(KNN)**\
   KNN adalah Algoritma yang menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru.

3. **Decision Tree**\
   Decision tree adalah model prediksi menggunakan struktur pohon atau struktur berhirarki.\
   Konsep dari pohon keputusan adalah mengubah data menjadi decision tree dan aturan-aturan keputusan.
   Kelebihan dari Decision Tree:\
   Kelebihan dari metode ini adalah mampu mengeliminasi perhitungan atau data-data yang kiranya tidak diperlukan. Sebab, sampel yang ada biasanya hanya diuji berdasarkan kriteria atau kelas tertentu saja.\
   salah satu kekurangan

4. **Boosting Algorithm(BaggingClassifier)**\
   Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner).

5. **RandomForestClassifier**\
   Random forest adalah kombinasi dari masing–masing tree yang baik kemudian dikombinasikan ke dalam satu model.

6. **XGBoost**\
   XGBoost atau eXtreme Gradient Boosting adalah algoritma berbasis pohon. XGBoost adalah bagian dari keluarga pohon (Decision tree, Random Forest, bagging, boosting, gradient boosting).

## Data Understanding

Untuk mengetahui apa saja variabel dan kegunaannya pada dataset kita dapat membaca dokumentasi dari penyedia dataset [Water Quality](https://www.kaggle.com/adityakadiwal/water-potability).

Dataset memiliki 3276 baris dan 10 kolom. Terdapat 9 kolom yang menampung data tentang sesuatu hal yang mempengaruhi air dan 1 kolom label yang berisikan bilangan biner yaitu 0 dan 1(0 adalah non-potable dan 1 adalah potable).

Deskripsi Variabel:

1. **pH**: indikator kondisi asam atau basa status air (0-14).
2. **Hardness**: Hardness adalah Kapasitas air untuk mengendapkan sabun dalam mg/L.
3. **Solids**: Total padatan terlarut dalam ppm.
4. **Chloramines**: Jumlah Kloramin dalam ppm.
5. **Sulfate**: Jumlah Sulfat yang dilarutkan dalam mg/L.
6. **Conductivity**: Konduktivitas listrik terhadap air dalam μS/cm..
7. **Organic_carbon**: Jumlah karbon organik dalam ppm.
8. **Trihalomethanes**: Jumlah Trihalomethanes dalam g/L.
9. **Turbidity**: Ukuran sifat pemancar cahaya terhadap air di NTU.
10. **Potability**: Menunjukkan apakah air aman untuk dikonsumsi manusia. Potable(1) dan Not potable(0)

langkah selanjutnya adalah memeriksa tipe data pada setiap variable dengan menggunakna method **info()**. untuk implementasi code sebagai berikut

```python
df.info()
```

yang akan menghasilkan keluaran seperti berikut

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3276 entries, 0 to 3275
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   ph               2785 non-null   float64
 1   Hardness         3276 non-null   float64
 2   Solids           3276 non-null   float64
 3   Chloramines      3276 non-null   float64
 4   Sulfate          2495 non-null   float64
 5   Conductivity     3276 non-null   float64
 6   Organic_carbon   3276 non-null   float64
 7   Trihalomethanes  3114 non-null   float64
 8   Turbidity        3276 non-null   float64
 9   Potability       3276 non-null   int64
dtypes: float64(9), int64(1)
memory usage: 256.1 KB
```

dengan begitu kita dapat mengetahui tipe data setiap variabel dataset.

Selanjutnya data di periksa terlebih dahulu apakah memiliki Missing Value atau tidak. untuk mengetahui apakah dataset memiliki missing value atau tidak. kita dapat menggunakan method **isnull** lalu menjumlahkannya dengan method **sum**. untuk penggunaannya seperti ini

```python
df.isnull().sum()
```

maka akan menghasilkan keluaran sebagai berikut

```python
ph                 491
Hardness             0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
Potability           0
dtype: int64
```

seperti yang bisa kita lihat terdapat missing value pada beberapa kolom pada dataset ini yaitu:

- pH: 491
- Sulfate: 781
- Trihalomethanes: 162

karena kualitas air adalah data yang sensitif kita tidak dapat mengubah data dengan memasukkan mean, median dan mode. Maka data yang mempunyai Missing Value dibuang. untuk membuang data yang mempunyai Missing Value kita bisa menggukan method **dropna**. untuk penggunaannya seperti ini

```python
df = df.dropna()
```

dengan begitu data baris atau rows yang memiliki data missing value akan dihapus atau dibuang

Langkah selanjutnya adalah memeriksa distribusi data label potable(1) dan non-potable(0). pada tahap ini saya mencoba memvisualisakannya menggunakan pie chart dengan menggunakan liblary dari **seaboarn** dan **matplotlib** karena mudah digunakan. dengan memvisualisasikannya kita dapat lebih mudah menganalisis distibusi data label. untuk penggunaanya dan hasilnya menjadi seperti ini

```python
colors = sns.color_palette('Set2')[0:6]
sns.palplot(colors)
labels = ['0: Non-Potable', '1: Potable']
data = [df['Potability'].value_counts()[0],
         df['Potability'].value_counts()[1]
        ]
fig1, ax1 = plt.subplots(figsize=[15,6])
ax1.pie(data, labels=labels, autopct='%1.1f%%',pctdistance=0.5, colors = colors)
plt.title("Water Potability", fontsize=20);
plt.show()
```

![pie chart 59.7% no-potable dan 40.3% potable](https://i.ibb.co/qWVsJsR/1.png)
bisa kita lihat bahwa distribusi label lebih banyak non-potable(0) dibandingkan dengan non-potable.

Untuk menyeimbangkan distribusi data saya menggunkan liblary dari sklearn yaitu **resample** dan **shuffle**

```python
zero  = df[df['Potability'] == 0]
one = df[df['Potability'] == 1]
df_minority_upsampled = resample(one, replace = True, n_samples = 1200)
df = pd.concat([zero, df_minority_upsampled])
df = shuffle(df)
```

dengan begitu distribusi data menjadi seimbang
![pie chart 50% no-potable dan 50% potable](https://i.ibb.co/TYnhbM2/3.png)

kemudian langkah selanjutnya adalah Exploratory Data Analysis Multivariate Analysis. pada tahap ini kita menganalisis korelasi antara dua variabel dengan fokus analisis pada variable Potability dengan variabel lainnya.

untuk melihat korelasi variable Potability dengan variabel lainnya dapa menggunakan menggunakan method **corr()** dari liblary **pandas**. untuk penggunaanya dan keluarnnya sebagai berikut:

```python
df.corr().abs()['Potability'].sort_values(ascending = False)
Potability         1.000000
Turbidity          0.039296
Organic_carbon     0.032550
Trihalomethanes    0.029843
Chloramines        0.028680
ph                 0.015669
Conductivity       0.011534
Hardness           0.008486
Solids             0.008273
Sulfate            0.000218
Name: Potability, dtype: float64
```

agar lebih mudah menganalisnya kita dapat memvisualisasikan korelasi antar variabel menggunakan liblary dari **seaborn heatmap**. dimana semakin cerah warnanya maka semakin berkorelasi antar variabelnya untuk penggunannya sebagai berikut

```python
import seaborn as sns
plt.figure(figsize = (15,9))
sns.heatmap(df.corr(), annot = True)
```

## Data Preparation

langkah pertama adalah Train-Test-Split. Membagi dataset menjadi data latih (train) dan data uji (test) proporsi pembagian data latih dan uji yang saya gunakan adalah 90:10. mengapa Train-Test-Split perlu dilakukan karena Kita harus mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Untuk membagi dataset menjadi data latih (train) dan data uji (test) saya menggunakan liblary **train_test_split** dari **sklearn** karena mudah digunakan. untuk untuk penggunannya sebagai berikut

```python
from sklearn.model_selection import train_test_split

X = diamonds.drop(["price"],axis =1)
y = diamonds["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)
```

langkah terakhir yang dilakukan untuk data preparation adalah standarisasi.
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Karena semua kolom berisikan nilai numerik maka saya menggunakan teknik StandarScaler.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, saya menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, saya melakukan standarisasi pada data uji.

## Modeling

Pada tahap modeling saya menggunakan bebera model untuk perbandingan model yang saya pakai adalah Logistic Regression, KNN, Decision Tree, Boosting Algorithm(BaggingClassifier), RandomForestClassifier dan XGBoost.

di modeling ini saya juga memakai RandomizedSearchCV dan GridSearchCV untuk mendapatkan tunning terbaik dalam model.

selanjutnya saya membandingkan akurasi prediksi tiap model dengan menggunakan data baru yaitu X_test sebagai ukuran keberhasilan model yang dibuat.

model dengan peforma terbaik adalah Random Forest dimana Random forest memiliki akurasi terhadap data baru sebesar 0.88.

## Evaluation

untuk proses evalusi model saya menggunakan matriks precision

**Precision** adalah kemampuan pengklasifikasi untuk tidak melabeli instance positif yang sebenarnya negatif. Untuk setiap kelas didefinisikan sebagai rasio positif benar dengan jumlah positif benar dan salah.

**Formula Precision**\
TP – True Positives\
FP – False Positives\
Precision – Accuracy of positive predictions.\
Precision = TP/(TP + FP)

Untuk menerapkan kode dan hasil dari evaluasi model yang sudah dibuat menjadi seperti ini

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_pred)
print(classification_report(y_true, y_pred))

              precision    recall  f1-score   support

           0       0.86      0.86      0.86       109
           1       0.89      0.89      0.89       131

    accuracy                           0.88       240
   macro avg       0.87      0.87      0.87       240
weighted avg       0.88      0.88      0.88       240
```

precision terhadap label 0 adalah 86% dan label 1 adalah 89%.
dengan pengertian bahwa model dapat memprediksi dengan benar terhadap label 0 sebanyak 86% dan terhadap label 1 sebanyak 89%.
