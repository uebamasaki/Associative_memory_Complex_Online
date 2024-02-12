import numpy as np
import random

# 定数
N = 100  # 素子数
P = 5  # パターンの数
SEED = 1  # シード値
phase = 20000  # フェーズ数

# 関数
def update_phase(W, C, random_index):
    """
    素子の状態を更新する関数
    """
    # 複素数で計算
    W[random_index] += np.dot(C[random_index], W)
    # 割る数が非常に小さい値でないかチェック
    mag = np.abs(W[random_index])
    if mag > 1e-10:
        W[random_index] /= mag
    else:
        W[random_index] = 1  # 小さい値で割る場合、1に固定
    # 正規化
    W[random_index] /= np.abs(W[random_index])

def connection_learning(Gusai, C):
    """
    結合を更新する関数
    """
    for mu in range(P):
        C += np.outer(Gusai[mu], Gusai[mu].conj())
    C /= (N - 1)
    np.fill_diagonal(C, 0)  # 自己結合を0に

def check(W, Gusai):
    """
    記憶を呼び出せたかチェックする関数
    """
    check = np.dot(Gusai, W.conj())
    return check.real / N, check.imag / N

def uniform():
    """
    一様分布からの乱数生成
    """
    return random.uniform(0, 1)

def file_output(recheck, imcheck, count):
    """
    想起度をファイルに出力する関数
    """
    for mu in range(P):
        filename = f"pattern{mu+1}.csv"
        mode = 'a' if count else 'w'
        with open(filename, mode) as f:
            if count == 0:
                f.write("Phase,Recall\n")
            recall = np.sqrt(recheck[mu]**2 + imcheck[mu]**2)
            f.write(f"{count+1},{recall}\n")

# 乱数シードの設定
random.seed(SEED)
np.random.seed(SEED)

# 初期化
W = np.zeros(N, dtype=np.complex_)  # 素子の状態
C = np.zeros((N, N), dtype=np.complex_)  # 結合
Gusai = np.zeros((P, N), dtype=np.complex_)  # 記憶パターン

# 記憶パターンの生成
for mu in range(P):
    random.seed(SEED + mu)
    for i in range(N):
        theta = 2.0 * np.pi * uniform()
        Gusai[mu, i] = np.cos(theta) + 1j * np.sin(theta)

# 学習フェーズ
connection_learning(Gusai, C)

# 想起フェーズ
for count in range(phase):
    random_index = random.randint(0, N - 1)  # 状態更新する素子をチョイス
    update_phase(W, C, random_index)  # 状態更新
    recheck, imcheck = check(W, Gusai)  # 想起度を計算
    file_output(recheck, imcheck, count)  # 想起度を出力
