"""classificarSensores2.py

LSTM sensor-type classifier: hyperparameter grid (sliding vs. tumbling windows)
and final configuration.

Protocol:
  - The first O readings of each data.txt column are split chronologically into
    training (70%), validation (15%), and test (15%) segments before windowing,
    so overlapping windows never appear in more than one subset.
  - Each segment is windowed with the step of the strategy (sliding = 1,
    tumbling = J). Each window yields four samples, one per sensor type.
  - A new model is built for each (O, J, strategy, seed).
  - Each model is trained once up to max(EPOCAS) and evaluated at the epoch
    checkpoints 100 to 500.

Modes (MODO):
  "grid"   runs the full grid with one seed.
  "final"  selects O and Ep on the validation accuracy of the grid (tumbling,
           J = 3, O in FINAL_FAIXA_O) and repeats that configuration with five
           seeds. Add "zscore_janela" to NORMALIZACOES for the per-window
           normalization ablation.

Outputs (in SAIDA):
  - runs.csv: one row per (configuration, seed, epoch checkpoint)
  - historico/<run>.csv: loss, accuracy, val_loss, val_accuracy and time per epoch
  - confusao/<run>_Ep<ep>.csv: 4x4 confusion matrix
  - predicoes/<run>_Ep<ep>.npz: y_true, y_pred, probabilities
  - modelos/<run>_Ep<ep>.h5 (optional)
  - metadados.json: experiment configuration and library versions
"""

import csv
import gc
import json
import os
import platform
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ARQUIVO_DADOS = "data.txt"

# "grid": full grid with one seed.
# "final": configuration selected on the grid validation accuracy, repeated with 5 seeds.
MODO = "grid"

# Output folders. PASTAS_VERSAO_1 lists folders used by the earlier script,
# which are never overwritten.
SAIDA_GRID = "Resultados2_grid"
SAIDA_FINAL = "Resultados2_final"
PASTAS_VERSAO_1 = {"Graficos", "."}
SAIDA = SAIDA_GRID                        # set in main() according to MODO

OBSERVACOES = [1000, 2000, 5000, 10000, 15000, 20000]
JANELAS = list(range(1, 11))
EPOCAS = [100, 200, 300, 400, 500]        # checkpoints evaluated within the same run
SEMENTES = [0]                            # grid: one seed

# Final mode: tumbling, J=3; O and Ep chosen by the highest grid val_acc.
FINAL_ESTRATEGIA = "tumbling"
FINAL_J = 3
FINAL_FAIXA_O = [10000, 15000, 20000]    # range of O considered in the final selection
SEMENTES_FINAL = [0, 1, 2, 3, 4]
ESTRATEGIAS = {1: "sliding", 2: "tumbling"}
NORMALIZACOES = ["nenhuma"]               # add "zscore_janela" for the ablation
FRACOES = (0.70, 0.15, 0.15)              # training, validation, test (chronological)
BATCH = 32
N_LATENCIA = 200                          # windows used to measure single-window latency

SALVAR_MODELOS = True
SALVAR_GRAFICOS = True
DETERMINISMO_ESTRITO = False              # True: more reproducible, slower

ROTULOS = ["Temperature", "Humidity", "Brightness", "Voltage"]
COLUNAS = [4, 5, 6, 7]                    # data.txt columns: temperature, humidity, light, voltage

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def carregar_series(caminho, n_max):
    """Read the first n_max lines of data.txt. Returns an array of shape (4, n)."""
    series = [[] for _ in COLUNAS]
    with open(caminho) as f:
        for i, linha in enumerate(f):
            if i == n_max:
                break
            partes = linha.split(" ")
            for k, c in enumerate(COLUNAS):
                series[k].append(float(partes[c]))
    return np.array(series, dtype=np.float32)


def dividir_cronologico(series, O):
    """Split the first O observations into contiguous training/validation/test blocks."""
    s = series[:, :O]
    fim_treino = int(O * FRACOES[0])
    fim_val = int(O * (FRACOES[0] + FRACOES[1]))
    return s[:, :fim_treino], s[:, fim_treino:fim_val], s[:, fim_val:]


def janelar(segmento, J, passo):
    """step=1 gives sliding windows; step=J gives tumbling windows (complete blocks,
        n // J windows). Each window yields 4 samples, one per class.
    """
    X, y = [], []
    n = segmento.shape[1]
    for i in range(0, n - J + 1, passo):
        for classe in range(segmento.shape[0]):
            X.append(segmento[classe, i:i + J])
            y.append(classe)
    return np.array(X, dtype=np.float32)[..., None], np.array(y, dtype=np.int64)


def normalizar(X, modo):
    """'zscore_janela' removes the absolute value range of each window (ablation).
        For J=1 every window becomes 0.
    """
    if modo == "nenhuma":
        return X
    if modo == "zscore_janela":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd_seguro = np.where(sd > 0, sd, 1.0)
        return np.where(sd > 0, (X - mu) / sd_seguro, 0.0).astype(np.float32)
    raise ValueError(modo)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def construir_modelo(J):
    """LSTM(128) followed by Dense(4, softmax). A new instance on every call."""
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(J, 1)),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


class Marcos(tf.keras.callbacks.Callback):
    """Records the time per epoch and triggers the evaluation at the EPOCAS checkpoints.
        Evaluation time is not counted as training time.
    """

    def __init__(self, marcos, ao_marcar):
        super().__init__()
        self.marcos = set(marcos)
        self.ao_marcar = ao_marcar
        self.tempo_treino = 0.0
        self.tempos_epoca = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.perf_counter() - self._t0
        self.tempo_treino += dt
        self.tempos_epoca.append(dt)
        ep = epoch + 1
        if ep in self.marcos:
            self.ao_marcar(ep, self.model, dict(logs or {}), self.tempo_treino)


def latencia_unitaria_ms(modelo, X, n):
    """Inference latency for one window at a time, as in the ClassifierAgent."""
    idx = np.linspace(0, len(X) - 1, num=min(n, len(X)), dtype=int)
    modelo(X[idx[:1]], training=False)                  # warm-up
    t0 = time.perf_counter()
    for i in idx:
        modelo(X[i:i + 1], training=False)
    return (time.perf_counter() - t0) / len(idx) * 1000.0

# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
CAMPOS = (["run_id", "normalizacao", "O", "J", "estrategia", "semente", "Ep",
           "n_treino", "n_val", "n_teste",
           "train_loss", "train_acc", "val_loss", "val_acc",
           "melhor_val_acc", "epoca_melhor_val",
           "test_loss", "test_acc", "test_f1_macro"]
          + [f"{m}_{r}" for r in ROTULOS for m in ("prec", "rec", "f1")]
          + ["tempo_treino_s", "latencia_ms", "n_parametros", "timestamp"])


def preparar_saida():
    for sub in ("historico", "confusao", "predicoes", "modelos", "graficos"):
        os.makedirs(os.path.join(SAIDA, sub), exist_ok=True)
    meta = {
        "observacoes": OBSERVACOES, "janelas": JANELAS, "epocas": EPOCAS,
        "sementes": SEMENTES, "estrategias": ESTRATEGIAS,
        "normalizacoes": NORMALIZACOES, "fracoes": FRACOES, "batch": BATCH,
        "modo": MODO,
        "split": "cronologico antes do janelamento; cada segmento janelado "
                 "com o passo da propria estrategia",
        "tensorflow": tf.__version__, "python": platform.python_version(),
        "plataforma": platform.platform(),
        "determinismo_estrito": DETERMINISMO_ESTRITO,
    }
    with open(os.path.join(SAIDA, "metadados.json"), "w") as f:
        json.dump(meta, f, indent=2)


def runs_concluidos():
    """Allows resuming the grid: a run counts only if it has all checkpoints."""
    caminho = os.path.join(SAIDA, "runs.csv")
    if not os.path.exists(caminho):
        return set()
    contagem = {}
    with open(caminho, newline="") as f:
        for linha in csv.DictReader(f):
            contagem[linha["run_id"]] = contagem.get(linha["run_id"], 0) + 1
    return {r for r, c in contagem.items() if c >= len(EPOCAS)}


def gravar_linha(linha):
    caminho = os.path.join(SAIDA, "runs.csv")
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        w.writerow(linha)


def gravar_historico(run_id, historico, tempos):
    caminho = os.path.join(SAIDA, "historico", f"{run_id}.csv")
    chaves = ["loss", "accuracy", "val_loss", "val_accuracy"]
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoca"] + chaves + ["tempo_epoca_s"])
        for e in range(len(historico["loss"])):
            w.writerow([e + 1] + [historico[k][e] for k in chaves] + [tempos[e]])


def grafico_historico(run_id, historico):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(historico["accuracy"], label="Training accuracy")
    plt.plot(historico["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(run_id)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAIDA, "graficos", f"{run_id}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def grafico_confusao(base, cm, nome_estrategia):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    cmap = plt.cm.Blues if nome_estrategia == "sliding" else plt.cm.Reds
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ROTULOS).plot(cmap=cmap)
    plt.title(base)
    plt.savefig(os.path.join(SAIDA, "graficos", f"{base}_confusao.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def executar_run(dados, O, J, id_estrategia, semente, norm):
    nome_estrategia = ESTRATEGIAS[id_estrategia]
    run_id = f"{norm}-O{O}-J{J}-S{id_estrategia}{nome_estrategia}-seed{semente}"

    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(semente)

    treino, val, teste = dividir_cronologico(dados, O)
    passo = 1 if nome_estrategia == "sliding" else J
    X_tr, y_tr = janelar(treino, J, passo)
    X_va, y_va = janelar(val, J, passo)       # same step as the strategy
    X_te, y_te = janelar(teste, J, passo)     # same step as the strategy
    X_tr, X_va, X_te = (normalizar(X, norm) for X in (X_tr, X_va, X_te))

    modelo = construir_modelo(J)
    n_param = modelo.count_params()
    melhor = {"acc": -1.0, "ep": 0}
    latencia = {"ms": None}
    linhas_run = []   # written to runs.csv only at the end of the run (resume without duplicates)

    def ao_marcar(ep, m, logs, tempo_treino):
        # best validation accuracy up to this checkpoint (from the partial history)
        vals = m.history.history.get("val_accuracy", []) + [logs["val_accuracy"]]
        i_melhor = int(np.argmax(vals))
        melhor["acc"], melhor["ep"] = float(vals[i_melhor]), i_melhor + 1

        prob = m.predict(X_te, batch_size=1024, verbose=0)
        y_pred = prob.argmax(axis=1)
        test_loss = float(tf.keras.losses.sparse_categorical_crossentropy(
            y_te, prob).numpy().mean())
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=[0, 1, 2, 3], zero_division=0)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2, 3])

        if ep == max(EPOCAS):
            latencia["ms"] = latencia_unitaria_ms(m, X_te, N_LATENCIA)

        base = f"{run_id}_Ep{ep}"
        np.savetxt(os.path.join(SAIDA, "confusao", f"{base}.csv"), cm,
                   fmt="%d", delimiter=",", header=",".join(ROTULOS))
        if SALVAR_GRAFICOS:
            grafico_confusao(base, cm, nome_estrategia)
        np.savez_compressed(os.path.join(SAIDA, "predicoes", f"{base}.npz"),
                            y_true=y_te, y_pred=y_pred, prob=prob)
        if SALVAR_MODELOS:
            m.save(os.path.join(SAIDA, "modelos", f"{base}.h5"))

        linha = {
            "run_id": run_id, "normalizacao": norm, "O": O, "J": J,
            "estrategia": nome_estrategia, "semente": semente, "Ep": ep,
            "n_treino": len(y_tr), "n_val": len(y_va), "n_teste": len(y_te),
            "train_loss": logs.get("loss"), "train_acc": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"), "val_acc": logs.get("val_accuracy"),
            "melhor_val_acc": melhor["acc"], "epoca_melhor_val": melhor["ep"],
            "test_loss": test_loss,
            "test_acc": accuracy_score(y_te, y_pred),
            "test_f1_macro": f1_score(y_te, y_pred, average="macro"),
            "tempo_treino_s": tempo_treino,
            "latencia_ms": latencia["ms"],
            "n_parametros": n_param,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for k, rot in enumerate(ROTULOS):
            linha[f"prec_{rot}"], linha[f"rec_{rot}"], linha[f"f1_{rot}"] = p[k], r[k], f1[k]
        linhas_run.append(linha)

    marcos = Marcos(EPOCAS, ao_marcar)
    hist = modelo.fit(X_tr, y_tr, epochs=max(EPOCAS), batch_size=BATCH,
                      validation_data=(X_va, y_va), shuffle=True,
                      callbacks=[marcos], verbose=0)

    gravar_historico(run_id, hist.history, marcos.tempos_epoca)
    if SALVAR_GRAFICOS:
        grafico_historico(run_id, hist.history)
    # Last step of the run: only now it counts as completed.
    for linha in linhas_run:
        gravar_linha(linha)
    print(f"{run_id}: concluido em {marcos.tempo_treino:.1f}s de treino")


def selecionar_final():
    """Read the grid runs.csv and choose O and Ep by the highest val_acc, restricted
        to tumbling windows, J=3 and O in FINAL_FAIXA_O. The test set is not used.
        Ties: smaller Ep, then smaller O.
    """
    caminho = os.path.join(SAIDA_GRID, "runs.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Rode o grid antes: {caminho} nao existe.")
    candidatos = []
    with open(caminho, newline="") as f:
        for l in csv.DictReader(f):
            if (l["normalizacao"] == "nenhuma" and l["estrategia"] == FINAL_ESTRATEGIA
                    and int(l["J"]) == FINAL_J and int(l["O"]) in FINAL_FAIXA_O):
                candidatos.append((-float(l["val_acc"]), int(l["Ep"]), int(l["O"])))
    if not candidatos:
        raise ValueError("Nenhuma configuracao candidata no grid.")
    neg_val, ep, O = min(candidatos)
    print(f"Selecionado pela validacao: O={O}, J={FINAL_J}, Ep={ep}, val_acc={-neg_val:.4f}")
    return O, ep


def main():
    global SAIDA, EPOCAS
    if DETERMINISMO_ESTRITO:
        tf.config.experimental.enable_op_determinism()

    if MODO == "grid":
        SAIDA = SAIDA_GRID
        observacoes, janelas, sementes = OBSERVACOES, JANELAS, SEMENTES
        estrategias = ESTRATEGIAS
    elif MODO == "final":
        O_final, ep_final = selecionar_final()
        SAIDA = SAIDA_FINAL
        EPOCAS = [ep_final]
        observacoes, janelas, sementes = [O_final], [FINAL_J], SEMENTES_FINAL
        estrategias = {k: v for k, v in ESTRATEGIAS.items() if v == FINAL_ESTRATEGIA}
    else:
        raise ValueError(MODO)

    if os.path.normpath(SAIDA) in PASTAS_VERSAO_1:
        raise ValueError(f"Pasta de saida '{SAIDA}' coincide com a da versao 1.")

    preparar_saida()
    dados = carregar_series(ARQUIVO_DADOS, max(observacoes))
    feitos = runs_concluidos()

    for norm in NORMALIZACOES:
        for O in observacoes:
            for J in janelas:
                if norm == "zscore_janela" and J == 1:
                    continue
                for semente in sementes:
                    for id_estrategia, nome in estrategias.items():
                        run_id = f"{norm}-O{O}-J{J}-S{id_estrategia}{nome}-seed{semente}"
                        if run_id in feitos:
                            continue
                        executar_run(dados, O, J, id_estrategia, semente, norm)


if __name__ == "__main__":
    main()
