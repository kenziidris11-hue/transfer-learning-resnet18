
import pandas as pd
import matplotlib.pyplot as plt

# Baca hasil perbandingan model
df = pd.read_csv("results/comparison.csv")

print("\n=== TABEL PERBANDINGAN MODEL ===")
print(df.to_string(index=False))

# Grafik akurasi validation
plt.figure(figsize=(9, 5))
plt.bar(df["mode"], df["best_val_accuracy"] * 100)
plt.ylabel("Akurasi Validation (%)")
plt.title("Perbandingan Akurasi Model")
plt.ylim(0, 105)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/comparison_accuracy.png")
plt.show()

# Grafik waktu training
plt.figure(figsize=(9, 5))
plt.bar(df["mode"], df["training_time_seconds"] / 60)
plt.ylabel("Waktu Training (menit)")
plt.title("Perbandingan Waktu Training")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/comparison_training_time.png")
plt.show()

# Grafik latency
plt.figure(figsize=(9, 5))
plt.bar(df["mode"], df["latency_ms"])
plt.ylabel("Latency CPU (ms/gambar)")
plt.title("Perbandingan Kecepatan Prediksi")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/comparison_latency.png")
plt.show()

print("\nGrafik tersimpan di folder results:")
print("- comparison_accuracy.png")
print("- comparison_training_time.png")
print("- comparison_latency.png")