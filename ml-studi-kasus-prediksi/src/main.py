from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data_mahasiswa.csv"
RESULTS = ROOT / "results"
FEATURES = ["ipk", "sks_lulus", "ulang_mk", "absensi_persen"]
TARGET = "status_tepat_waktu"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = ["id", *FEATURES, TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    df = df.drop_duplicates(subset="id").dropna(subset=required)
    return df


def run():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    RESULTS.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)
    (RESULTS / "classification_report.txt").write_text(
        f"Jumlah data: {len(df)}\nAkurasi data uji: {accuracy:.3f}\n\n{report}",
        encoding="utf-8",
    )

    output = X_test.copy()
    output.insert(0, "id", df.loc[X_test.index, "id"])
    output["label_asli"] = y_test
    output["prediksi_model"] = pred
    output.to_csv(RESULTS / "hasil_prediksi.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix - Dataset Simulasi")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    new_student = pd.DataFrame([{
        "ipk": 3.2, "sks_lulus": 100, "ulang_mk": 2, "absensi_persen": 84
    }])
    new_prediction = model.predict(new_student)[0]

    print("=== Studi Kasus Machine Learning ===")
    print(f"Jumlah data setelah pembersihan: {len(df)}")
    print(f"Akurasi data uji: {accuracy:.3f}")
    print("\nLaporan klasifikasi:")
    print(report)
    print(f"Contoh prediksi data baru: {new_prediction}")
    print(f"Hasil disimpan di folder: {RESULTS}")
    print("Catatan: seluruh dataset merupakan simulasi pembelajaran.")


if __name__ == "__main__":
    run()
