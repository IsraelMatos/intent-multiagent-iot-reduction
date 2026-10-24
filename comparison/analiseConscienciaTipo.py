"""analiseConscienciaTipo.py

Two analyses complementary to compararReducao.py, without the perceptron (runs in seconds):

(a) Per-type policy vs. uniform ar, with the same recovery (linear interpolation, M2)
    and the same mean total traffic.
(b) DF and DP with thresholds scaled by the standard deviation of each sensor type
    (requires the type) vs. a single absolute threshold for all sensors, at the
    same mean total traffic.

Run in the same folder as compararReducao.py (uses data.txt and the same functions).
Output: Resultados_comparacao/consciencia_tipo_curvas.csv and _comparacao.csv
"""

import csv
import os

import numpy as np

from compararReducao import (DISPOSITIVOS, HEARTBEAT, MAX_LINHAS, N_OBS, POLITICAS, SAIDA,
                             ARQUIVO_DADOS, carregar_colunas, metricas, recuperar_linear_pipeline,
                             reducao)
from leclerc_metodos import df_leclerc, dp_leclerc

AR_UNIFORME = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60]
FRACOES = np.logspace(-3, 0.5, num=15)          # per-type threshold: fraction of the standard deviation
ABSOLUTOS = np.logspace(-4, 3, num=36)          # single threshold for all sensors, in raw units
ALVOS = [80.0, 82.74, 85.0, 90.0, 91.26, 92.5, 95.0]   # 82.74 and 91.26: mean reductions of Conservative and Aggressive (M1/M2)


def media_por_config(segmentos, funcao):
    """Apply funcao(x, tipo) to the 8 devices; return the mean reduction and mean NRMSE/MAE."""
    reds, nrs, mas = [], [], []
    for rotulo, _, tipo, _ in DISPOSITIVOS:
        x = segmentos[rotulo]
        n_tx, rec = funcao(x, tipo)
        m = metricas(x, rec)
        reds.append(reducao(n_tx))
        nrs.append(m["nrmse_pct"])
        mas.append(m["mae"])
    return float(np.mean(reds)), float(np.mean(nrs)), float(np.mean(mas))


def interpolar(curva, alvo):
    """curva: list of (reduction, nrmse, mae). Interpolates at alvo; NaN outside the range."""
    pts = sorted(curva)
    red = np.array([p[0] for p in pts])
    if not (red.min() <= alvo <= red.max()):
        return float("nan"), float("nan")
    red_u, idx = np.unique(red, return_index=True)
    nr = np.array([p[1] for p in pts])[idx]
    ma = np.array([p[2] for p in pts])[idx]
    return float(np.interp(alvo, red_u, nr)), float(np.interp(alvo, red_u, ma))


def main():
    cols = carregar_colunas(ARQUIVO_DADOS, MAX_LINHAS)
    desvios = {t: float(np.std(v)) for t, v in cols.items()}
    segmentos = {r: cols[t][p:p + N_OBS] for r, _, t, p in DISPOSITIVOS}

    curvas, linhas_curva = {}, []

    def registrar(variante, parametro, resultado):
        curvas.setdefault(variante, []).append(resultado)
        linhas_curva.append({"variante": variante, "parametro": parametro,
                             "reducao_media_pct": resultado[0], "nrmse_medio_pct": resultado[1],
                             "mae_medio": resultado[2]})

    # (a) same recovery as M2: per-type policy vs. uniform ar
    for cenario, pol in POLITICAS.items():
        registrar("M2_politica", cenario,
                  media_por_config(segmentos, lambda x, t: recuperar_linear_pipeline(x, pol[t][0])))
    for ar in AR_UNIFORME:
        registrar("M2_ar_uniforme", ar,
                  media_por_config(segmentos, lambda x, t: recuperar_linear_pipeline(x, ar)))

    # (b) per-type scaled threshold vs. single absolute threshold
    for f in FRACOES:
        registrar("DF_por_tipo", f, media_por_config(
            segmentos, lambda x, t: df_leclerc(x, f * desvios[t], max_interval=HEARTBEAT)))
        registrar("DP_por_tipo", f, media_por_config(
            segmentos, lambda x, t: dp_leclerc(x, f * desvios[t], max_interval=HEARTBEAT)))
    for d in ABSOLUTOS:
        registrar("DF_limiar_unico", d, media_por_config(
            segmentos, lambda x, t: df_leclerc(x, d, max_interval=HEARTBEAT)))
        registrar("DP_limiar_unico", d, media_por_config(
            segmentos, lambda x, t: dp_leclerc(x, d, max_interval=HEARTBEAT)))

    comparacao = []
    for alvo in ALVOS:
        linha = {"reducao_media_alvo_pct": alvo}
        for variante in ("M2_ar_uniforme", "DF_por_tipo", "DF_limiar_unico",
                         "DP_por_tipo", "DP_limiar_unico"):
            nr, ma = interpolar(curvas[variante], alvo)
            linha[f"{variante}_nrmse"] = nr
            linha[f"{variante}_mae"] = ma
        comparacao.append(linha)

    os.makedirs(SAIDA, exist_ok=True)
    for nome, dados in (("consciencia_tipo_curvas.csv", linhas_curva),
                        ("consciencia_tipo_comparacao.csv", comparacao)):
        with open(os.path.join(SAIDA, nome), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dados[0].keys()))
            w.writeheader()
            w.writerows(dados)

    print("(a) Politica por tipo x ar uniforme (mesma recuperacao linear):")
    for red, nr, ma in curvas["M2_politica"]:
        u_nr, u_ma = interpolar(curvas["M2_ar_uniforme"], red)
        print(f"  reducao {red:6.2f}%  politica NRMSE {nr:.3f} MAE {ma:.3f} | "
              f"uniforme NRMSE {u_nr:.3f} MAE {u_ma:.3f}")
    print("\n(b) Limiar por tipo x limiar unico, mesma reducao media (NRMSE %):")
    for linha in comparacao:
        print(f"  {linha['reducao_media_alvo_pct']:6.2f}%  "
              f"DF por tipo {linha['DF_por_tipo_nrmse']:.3f} x unico {linha['DF_limiar_unico_nrmse']:.3f} | "
              f"DP por tipo {linha['DP_por_tipo_nrmse']:.3f} x unico {linha['DP_limiar_unico_nrmse']:.3f}")


if __name__ == "__main__":
    main()
