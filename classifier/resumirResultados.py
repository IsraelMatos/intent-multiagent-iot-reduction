"""
resumirResultados.py

Summarizes the outputs of classificarSensores2.py into the values reported in
Section 6.1 of the paper: descriptive statistics per strategy (Table 4), ANOVA
between strategies, training time ratio, mean accuracy per J and per epoch
checkpoint, per-class metrics of the selected model (Table 5), and the
per-window normalization ablation.

Usage (in the folder that contains Resultados2_grid and Resultados2_final):
    python resumirResultados.py
"""

import csv
import os

import numpy as np
from scipy import stats

GRID = os.path.join("Resultados2_grid", "runs.csv")
FINAL = os.path.join("Resultados2_final", "runs.csv")
ROTULOS = ["Temperature", "Humidity", "Brightness", "Voltage"]


def ler(caminho):
    with open(caminho, newline="") as f:
        return list(csv.DictReader(f))


def valores(linhas, campo, **filtro):
    return np.array([float(l[campo]) for l in linhas
                     if all(l[k] == str(v) for k, v in filtro.items())])


def estatisticas(x):
    return {"count": len(x), "mean": x.mean(), "std": x.std(ddof=1), "min": x.min(),
            "p25": np.percentile(x, 25), "median": np.median(x),
            "p75": np.percentile(x, 75), "max": x.max()}


def main():
    grid = ler(GRID)

    print("Table 4: test accuracy per windowing strategy (grid)")
    acc = {e: valores(grid, "test_acc", estrategia=e) for e in ("sliding", "tumbling")}
    for e, x in acc.items():
        print(f"  {e:9s} " + "  ".join(f"{k} {v:.3f}" if k != "count" else f"{k} {v}"
                                        for k, v in estatisticas(x).items()))
    f, p = stats.f_oneway(acc["sliding"], acc["tumbling"])
    print(f"  ANOVA: F = {f:.3f}, p = {p:.3f}")

    # training time at the last checkpoint is the total training time of each run
    ultimo = max(int(l["Ep"]) for l in grid)
    tempo = {e: valores(grid, "tempo_treino_s", estrategia=e, Ep=ultimo).sum()
             for e in ("sliding", "tumbling")}
    print(f"  Total training time, sliding / tumbling: {tempo['sliding'] / tempo['tumbling']:.1f}x")

    print("\nMean test accuracy per window size J and per epoch checkpoint")
    for campo in ("J", "Ep"):
        niveis = sorted({int(l[campo]) for l in grid})
        medias = {n: valores(grid, "test_acc", **{campo: n}).mean() for n in niveis}
        print(f"  {campo}: " + ", ".join(f"{n}: {m:.3f}" for n, m in medias.items()))

    final = ler(FINAL)
    print("\nTable 5: selected model, mean +- std over seeds")
    for norm in sorted({l["normalizacao"] for l in final}):
        linhas = [l for l in final if l["normalizacao"] == norm]
        teste = valores(linhas, "test_acc")
        treino = valores(linhas, "train_acc")
        print(f"  normalization '{norm}' ({len(linhas)} seeds): test accuracy "
              f"{teste.mean():.3f} +- {teste.std(ddof=1):.3f}, training accuracy {treino.mean():.3f}")
        if norm != "nenhuma":
            continue
        for r in ROTULOS:
            partes = []
            for m in ("prec", "rec", "f1"):
                x = valores(linhas, f"{m}_{r}")
                partes.append(f"{m} {x.mean():.3f} +- {x.std(ddof=1):.3f}")
            print(f"    {r:12s} " + "  ".join(partes))


if __name__ == "__main__":
    main()
