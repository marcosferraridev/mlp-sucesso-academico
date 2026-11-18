"""Treina e avalia a MLP no dataset de sucesso acadêmico."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from preprocessamento import NOMES_CLASSES, SEED, carregar_dados, carregar_pesos_classe


RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
GRAFICOS = RESULTADOS / "graficos"


def construir_mlp(input_dim, ativacao, learning_rate):
    tf.keras.utils.set_random_seed(SEED)
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation=ativacao),
        tf.keras.layers.Dense(32, activation=ativacao),
        tf.keras.layers.Dense(16, activation=ativacao),
        tf.keras.layers.Dense(len(NOMES_CLASSES), activation="softmax"),
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def treinar(modelo, dados, epocas, batch_size, class_weight=None):
    X_train, X_val, _, y_train, y_val, _ = dados
    rng = np.random.default_rng(SEED)
    alvos = np.eye(len(NOMES_CLASSES))[y_train]
    historico = {chave: [] for chave in ("loss", "val_loss", "accuracy", "val_accuracy")}
    for epoca in range(1, epocas + 1):
        indices = rng.permutation(len(y_train))
        for lote, inicio in enumerate(range(0, len(indices), batch_size), 1):
            idx = indices[inicio:inicio + batch_size]
            X_lote, y_lote = X_train[idx], y_train[idx]
            modelo.train_on_batch(X_lote, y_lote, class_weight=class_weight)
            probabilidades = modelo(X_lote, training=False).numpy()
            erros = np.abs(alvos[idx] - probabilidades).mean(axis=0)
            print(f"Época {epoca}/{epocas}, lote {lote}, erros por classe: {erros.round(4)}")

        # Mede a época inteira com os pesos atualizados e sem ponderar as métricas.
        treino = modelo.test_on_batch(X_train, y_train, return_dict=True)
        validacao = modelo.test_on_batch(X_val, y_val, return_dict=True)
        for chave in ("loss", "accuracy"):
            historico[chave].append(float(treino[chave]))
            historico[f"val_{chave}"].append(float(validacao[chave]))
        print(f"Época {epoca}: perda treino/validação "
              f"{treino['loss']:.4f}/{validacao['loss']:.4f}")
    print("Vetor de perdas de validação por época:", np.round(historico["val_loss"], 4))
    return historico


def avaliar(modelo, X, y):
    previstos = modelo(X, training=False).numpy().argmax(axis=1)
    classes = range(len(NOMES_CLASSES))
    metricas = classification_report(
        y, previstos, labels=classes, target_names=NOMES_CLASSES,
        output_dict=True, zero_division=0,
    )
    return {
        "acuracia": metricas["accuracy"],
        "f1_macro": metricas["macro avg"]["f1-score"],
        "por_classe": {nome: metricas[nome] for nome in NOMES_CLASSES},
        "matriz_confusao": confusion_matrix(y, previstos, labels=classes),
    }


def main():
    tf.config.experimental.enable_op_determinism()
    dados = carregar_dados()
    X_train, X_val, X_test, _, y_val, y_test = dados
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    pesos = carregar_pesos_classe()
    registros = []
    melhor_modelo = None
    melhor_f1 = -1
    melhor_historico = None

    def executar(nome, ativacao, learning_rate, epocas, batch_size, ponderado):
        nonlocal melhor_modelo, melhor_f1, melhor_historico
        print(f"\nExperimento: {nome}")
        modelo = construir_mlp(X_train.shape[1], ativacao, learning_rate)
        hist = treinar(modelo, dados, epocas, batch_size, pesos if ponderado else None)
        metricas = avaliar(modelo, X_val, y_val)
        registro = dict(
            experimento=nome, ativacao=ativacao, learning_rate=learning_rate,
            epocas=epocas, batch_size=batch_size, class_weight=ponderado,
            loss_treino=hist["loss"][-1], loss_val=hist["val_loss"][-1],
            acc_val=metricas["acuracia"], f1_macro_val=metricas["f1_macro"],
        )
        registros.append(registro)
        if registro["f1_macro_val"] > melhor_f1:
            melhor_modelo, melhor_historico = modelo, hist
            melhor_f1 = registro["f1_macro_val"]
        return registro

    relu = executar("ReLU", "relu", 0.001, 20, 32, False)
    tanh = executar("Tanh", "tanh", 0.001, 20, 32, False)
    ativacao = max((relu, tanh), key=lambda r: r["f1_macro_val"])["ativacao"]
    executar("Taxa 0.0003", ativacao, 0.0003, 20, 32, False)
    executar("30 épocas", ativacao, 0.001, 30, 32, False)
    executar("Lote 64", ativacao, 0.001, 20, 64, False)
    executar("Pesos de classe", ativacao, 0.001, 20, 32, True)
    tabela = pd.DataFrame(registros)
    tabela.to_csv(RESULTADOS / "metricas.csv", index=False)

    teste = avaliar(melhor_modelo, X_test, y_test)
    linhas = [
        {"classe": nome, "precisao": m["precision"], "recall": m["recall"],
         "f1": m["f1-score"], "amostras": int(m["support"])}
        for nome, m in teste["por_classe"].items()
    ]
    linhas.append({"classe": "Geral", "acuracia": teste["acuracia"], "f1": teste["f1_macro"]})
    pd.DataFrame(linhas).to_csv(RESULTADOS / "metricas_teste.csv", index=False)
    print("Matriz de confusão no teste:\n", teste["matriz_confusao"])

    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for eixo, chave, titulo in zip(eixos, ("loss", "accuracy"), ("Perda", "Acurácia")):
        epocas = range(1, len(melhor_historico[chave]) + 1)
        eixo.plot(epocas, melhor_historico[chave], label="Treino")
        eixo.plot(epocas, melhor_historico[f"val_{chave}"], label="Validação")
        eixo.set(xlabel="Época", ylabel=titulo)
        eixo.legend()
    fig.savefig(GRAFICOS / "desempenho.png", dpi=160)
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(7, 3.5), layout="constrained")
    barras = eixo.barh(tabela["experimento"], tabela["f1_macro_val"])
    eixo.bar_label(barras, fmt="%.3f", padding=4)
    eixo.invert_yaxis()
    eixo.set(xlabel="F1 macro na validação", xlim=(0, 1))
    fig.savefig(GRAFICOS / "experimentos.png", dpi=160)
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(5, 4), layout="constrained")
    ConfusionMatrixDisplay(teste["matriz_confusao"], display_labels=NOMES_CLASSES).plot(
        ax=eixo, cmap="Blues", colorbar=False,
    )
    eixo.set(xlabel="Classe prevista", ylabel="Classe real")
    fig.savefig(GRAFICOS / "matriz_confusao.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
