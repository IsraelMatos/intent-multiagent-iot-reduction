"""converterClassificador.py

Converts the final classifier (trained with TensorFlow 2.17 / Keras 3) into the
format loaded by the pipeline (TensorFlow 2.15 / Keras 2), without retraining.
The pipeline loads it with tf.keras.models.load_model("model_classifier.h5").

Two steps, each in its own environment:

  1) In the classifier environment (TF 2.17), in the folder that contains Resultados2_final:
       python converterClassificador.py exportar
     Writes classificador_seed4.npz with the weights and a reference set
     (input windows and the probabilities predicted by the original model).

  2) In the pipeline environment (TF 2.15), in the pipeline folder, with the .npz copied:
       python converterClassificador.py gerar
     Rebuilds the same architecture (LSTM 128 + Dense 4 softmax, input 3x1),
     loads the weights, checks that the probabilities match those of the
     original model, and writes model_classifier.h5. An existing
     model_classifier.h5 is kept as model_classifier_v1.h5.
"""

import os
import shutil
import sys

import numpy as np
import tensorflow as tf

ORIGEM = os.path.join("Resultados2_final", "modelos",
                      "nenhuma-O15000-J3-S2tumbling-seed4_Ep400.h5")
NPZ = "classificador_seed4.npz"
DESTINO = "model_classifier.h5"
BACKUP = "model_classifier_v1.h5"
J = 3
TOLERANCIA = 1e-5


def janelas_referencia():
    """Fixed windows covering the ranges of the four sensor types and intermediate values."""
    rng = np.random.default_rng(0)
    faixas = [(15, 30), (25, 45), (0, 700), (2.3, 2.8), (0, 100)]
    X = np.concatenate([rng.uniform(lo, hi, size=(100, J)) for lo, hi in faixas])
    return X.astype(np.float32)[..., None]


def exportar():
    modelo = tf.keras.models.load_model(ORIGEM, compile=False)
    X = janelas_referencia()
    prob = modelo.predict(X, verbose=0)
    pesos = modelo.get_weights()
    np.savez(NPZ, X=X, prob=prob, n_pesos=len(pesos),
             **{f"w{i}": w for i, w in enumerate(pesos)})
    print(f"TF {tf.__version__}: {len(pesos)} tensores de peso exportados para {NPZ}")
    print("Formatos:", [w.shape for w in pesos])


def gerar():
    dados = np.load(NPZ)
    pesos = [dados[f"w{i}"] for i in range(int(dados["n_pesos"]))]
    modelo = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(J, 1)),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    modelo.set_weights(pesos)
    prob = modelo.predict(dados["X"], verbose=0)
    dif = float(np.max(np.abs(prob - dados["prob"])))
    iguais = int((prob.argmax(1) == dados["prob"].argmax(1)).sum())
    print(f"TF {tf.__version__}: diferenca maxima de probabilidade {dif:.2e}; "
          f"classes iguais em {iguais}/{len(prob)} janelas")
    if dif > TOLERANCIA:
        sys.exit("Diferenca acima da tolerancia; model_classifier.h5 NAO foi gravado.")
    if os.path.exists(DESTINO) and not os.path.exists(BACKUP):
        shutil.copy2(DESTINO, BACKUP)
        print(f"Modelo anterior preservado em {BACKUP}")
    modelo.save(DESTINO)
    recarregado = tf.keras.models.load_model(DESTINO)
    dif2 = float(np.max(np.abs(recarregado.predict(dados["X"], verbose=0) - dados["prob"])))
    print(f"{DESTINO} gravado e recarregado com load_model; diferenca maxima {dif2:.2e}")


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else ""
    if etapa == "exportar":
        exportar()
    elif etapa == "gerar":
        gerar()
    else:
        sys.exit("Uso: python converterClassificador.py exportar | gerar")
