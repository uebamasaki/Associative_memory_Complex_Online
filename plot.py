import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import japanize_matplotlib

N = 5

# 日本語フォントの準備
fp = FontProperties(fname=r'/Library/Fonts/Arial Unicode.ttf', size=11)

# PDF出力準備
pdf = PdfPages("graph.pdf")

plt.figure(figsize=(10, 6))

# ファイル名リスト
labels = [f"pattern{i+1}" for i in range(N)]

# プロット
for i, label in enumerate(labels, start=1):
    df = pd.read_csv(f"{label}.csv")
    plt.plot(df["Phase"], df["Recall"], label=label)

plt.title("")
plt.xlabel("Phase")
plt.ylabel("想起度")
plt.legend()

# グラフをPDFに保存
pdf.savefig()
pdf.close()

plt.show()
