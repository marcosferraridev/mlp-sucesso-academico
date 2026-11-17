"""Pré-processamento do dataset "Predict Students' Dropout and Academic Success" (UCI 697).

Card A2 (limpeza e codificação) + Card A3 (normalização, divisão e funções públicas).
Não usa TensorFlow/PyTorch, não treina nada e não usa o conjunto de teste para decisões.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42  # seed fixa em tudo, para reprodutibilidade
CAMINHO_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "data.csv"

NOMES_CLASSES = ["Dropout", "Enrolled", "Graduate"]  # índice = código da classe (0, 1, 2)

# Colunas que são CÓDIGOS inteiros sem ordem (curso, modo de candidatura etc.).
# Viram categorias (one-hot), e não números, para a rede não inferir uma ordem
# que não existe. Os nomes seguem o arquivo original (inclusive "Nacionality").
COLUNAS_CATEGORICAS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Atributos descartados. Vazio por padrão: todos os 36 atributos são usados.
# Se o grupo decidir descartar algum após a EDA (ver docs/dataset.md), liste aqui.
COLUNAS_DESCARTADAS = []

# Categorias com frequência menor que isto no TREINO viram "outros"
# (evita colunas one-hot quase vazias).
FREQ_MINIMA_CATEGORIA = 0.01


def _ler_csv() -> pd.DataFrame:
    """Lê o CSV (separador ';') e limpa os nomes das colunas (espaços, tabs)."""
    if not CAMINHO_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CAMINHO_CSV}. Baixe o data.csv em "
            "https://archive.ics.uci.edu/dataset/697 e salve em data/raw/."
        )
    df = pd.read_csv(CAMINHO_CSV, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def _separar_x_y(df: pd.DataFrame):
    """Separa atributos e target; converte o target de texto para 0/1/2."""
    if "Target" not in df.columns:
        raise KeyError(f"Coluna 'Target' não encontrada. Colunas lidas: {list(df.columns)}")
    desconhecidas = set(df["Target"].unique()) - set(NOMES_CLASSES)
    if desconhecidas:
        raise ValueError(f"Classes inesperadas no Target: {desconhecidas}")
    y = df["Target"].map({nome: i for i, nome in enumerate(NOMES_CLASSES)}).astype(int)
    X = df.drop(columns=["Target"] + COLUNAS_DESCARTADAS)
    return X, y


def _colunas_categoricas(X: pd.DataFrame) -> list:
    """Categóricas presentes no arquivo; falha com mensagem clara se algum nome não bater."""
    faltando = [
        c for c in COLUNAS_CATEGORICAS
        if c not in X.columns and c not in COLUNAS_DESCARTADAS
    ]
    if faltando:
        raise KeyError(
            f"Colunas categóricas não encontradas: {faltando}. Colunas lidas: {list(X.columns)}"
        )
    return [c for c in COLUNAS_CATEGORICAS if c in X.columns]


def _ajustar_regras(X_train: pd.DataFrame, categoricas: list) -> dict:
    """Aprende, SOMENTE no treino, quais categorias são mantidas (freq >= mínimo)."""
    mantidas = {}
    for col in categoricas:
        freq = X_train[col].value_counts(normalize=True)
        mantidas[col] = sorted(freq[freq >= FREQ_MINIMA_CATEGORIA].index.tolist())
    return mantidas


def _codificar(X: pd.DataFrame, categoricas: list, mantidas: dict) -> pd.DataFrame:
    """One-hot com categorias fixas (as do treino). Raras ou nunca vistas viram 'outros'."""
    X = X.copy()
    for col in categoricas:
        valores = X[col].where(X[col].isin(mantidas[col]), other=-1)  # -1 = "outros"
        X[col] = pd.Categorical(valores, categories=mantidas[col] + [-1])
    return pd.get_dummies(X, columns=categoricas, dtype=float)


@lru_cache(maxsize=1)
def _preparar():
    """Roda o pipeline uma única vez (mesmo resultado a cada chamada)."""
    X, y = _separar_x_y(_ler_csv())
    if X.isna().any().any():
        raise ValueError("Há valores ausentes (NaN) no arquivo; trate-os antes de continuar.")
    categoricas = _colunas_categoricas(X)

    # Divisão estratificada 70/15/15: separa 70% de treino e divide os 30% restantes ao meio.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    # Tudo que "aprende" dos dados (categorias mantidas, média, desvio) usa só o treino.
    mantidas = _ajustar_regras(X_tr, categoricas)
    X_tr, X_va, X_te = (_codificar(p, categoricas, mantidas) for p in (X_tr, X_va, X_te))

    # Padronização (média 0, desvio 1) só nas colunas numéricas não binárias.
    # As colunas one-hot e as binárias (0/1) ficam como estão.
    numericas = [c for c in X_tr.columns if X_tr[c].nunique() > 2]
    scaler = StandardScaler().fit(X_tr[numericas])
    for parte in (X_tr, X_va, X_te):
        parte[numericas] = scaler.transform(parte[numericas])

    arrays = [p.to_numpy(dtype=np.float32) for p in (X_tr, X_va, X_te)]
    ys = [p.to_numpy(dtype=np.int64) for p in (y_tr, y_va, y_te)]
    if any(np.isnan(a).any() for a in arrays):
        raise ValueError("NaN encontrado em X após o pré-processamento.")
    return (*arrays, *ys)


def carregar_dados() -> tuple:
    """Retorna X_train, X_val, X_test, y_train, y_val, y_test (arrays NumPy)."""
    return _preparar()


def carregar_pesos_classe() -> dict:
    """Pesos de classe calculados só com o treino, para usar em class_weight (Keras)."""
    y_train = _preparar()[3]
    classes = np.arange(len(NOMES_CLASSES))
    pesos = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(p) for c, p in zip(classes, pesos)}


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = carregar_dados()
    print("Shapes:")
    for nome, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        print(f"  {nome}: {X.shape}")
    for nome, y in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        print(f"  {nome}: {y.shape}")
    print(f"\nNº de atributos de entrada (após one-hot): {X_train.shape[1]}")
    print("\nDistribuição das classes por partição:")
    for nome, y in [("treino", y_train), ("validação", y_val), ("teste", y_test)]:
        contagem = np.bincount(y, minlength=len(NOMES_CLASSES))
        partes = ", ".join(
            f"{NOMES_CLASSES[i]}={contagem[i]} ({contagem[i] / len(y):.1%})"
            for i in range(len(NOMES_CLASSES))
        )
        print(f"  {nome}: {partes}")
    print("\nPesos de classe (treino):", carregar_pesos_classe())
