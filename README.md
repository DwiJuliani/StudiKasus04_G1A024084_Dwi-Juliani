# Studi Kasus Materi 4: Prediksi Kelulusan Mahasiswa Menggunakan Machine Learning

## Latar Belakang
Machine Learning dapat mempelajari pola dari data untuk membantu membuat prediksi. Dalam studi kasus ini, model memprediksi apakah seorang mahasiswa berisiko terlambat lulus berdasarkan data akademik sederhana.

## Tujuan
- Memahami alur kerja Machine Learning dari data hingga evaluasi.
- Membersihkan dan memeriksa dataset.
- Melatih model klasifikasi menggunakan Supervised Learning.
- Mengevaluasi hasil model pada data uji.
- Memprediksi data mahasiswa baru sebagai demonstrasi.

## Dataset
File `data/data_mahasiswa.csv` berisi data **simulasi**, bukan data mahasiswa nyata. Fitur yang digunakan:
- `ipk`: indeks prestasi kumulatif simulasi.
- `sks_lulus`: jumlah SKS yang telah diselesaikan.
- `ulang_mk`: jumlah mata kuliah yang diulang.
- `absensi_persen`: persentase kehadiran.
- `status_tepat_waktu`: label target (`Ya` atau `Tidak`).

## Struktur Folder
```text
ml-studi-kasus-prediksi/
├── data/
│   └── data_mahasiswa.csv
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── results/
├── README.md
└── requirements.txt
```

## Instalasi
Pastikan Python 3.9 atau lebih baru sudah tersedia. Jalankan dari folder proyek:

```bash
pip install -r requirements.txt
```

## Menjalankan Program
```bash
python src/main.py
```

Program menampilkan akurasi dan laporan klasifikasi di terminal, lalu menyimpan:
- `results/hasil_prediksi.csv`
- `results/confusion_matrix.png`

## Menjalankan Pengujian
```bash
python -m unittest discover -s tests
```

## Alur Machine Learning
1. Membaca dan memeriksa dataset.
2. Memisahkan fitur dan label.
3. Membagi data menjadi data latih dan data uji.
4. Melatih model Decision Tree.
5. Mengukur performa pada data uji.
6. Menggunakan model untuk memprediksi contoh data baru.

## Batasan
Dataset sengaja dibuat untuk pembelajaran. Hasil model tidak boleh dipakai untuk menilai atau mengambil keputusan nyata tentang mahasiswa. Pada kasus nyata, perlu data yang representatif, persetujuan dan perlindungan privasi, pemeriksaan bias, serta evaluasi yang lebih menyeluruh.
