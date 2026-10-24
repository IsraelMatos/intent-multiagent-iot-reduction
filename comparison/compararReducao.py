"""compararReducao.py

Offline comparison between the edge reduction with cloud recovery of the pipeline
and three edge data reduction methods of Leclerc et al. (2026), on the same eight
device segments of the functional experiment.

Methods:
  M1  Pipeline: block mean with ar per sensor type (Table 2) + perceptron recovery
  M2  Pipeline with linear recovery: same transmitted values as M1 (means + seed),
      linear interpolation with each mean at the center of its block
  M3  DA (Leclerc et al.), uniform N
  M4  DF (Leclerc et al.), threshold delta = fraction of the standard deviation of the type
  M5  DP (Leclerc et al.), margin eps = fraction of the standard deviation of the type

Step 0: M1 must reproduce the MAE and MSE in mqtt_pipeline_results_v11.json;
otherwise the script stops before the comparisons.

Run in the pipeline environment (TF 2.15), in a folder with data.txt, the 12
perceptron .h5 models, the v11 results JSON, and leclerc_metodos.py.
"""

import csv
import json
import os
import sys
import time

import numpy as np

from leclerc_metodos import da_leclerc, df_leclerc, dp_leclerc

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ARQUIVO_DADOS = "data.txt"
PASTA_MODELOS = "."
JSON_V11 = "mqtt_pipeline_results_v11.json"
SAIDA = "Resultados_comparacao"

N_OBS = 1000                      # NUM_OBSERVATIONS of the pipeline
PERCEPTRON_J = 10                 # seed values sent in the reconstruction package
MAX_LINHAS = 22000                # MAX_POSITION of sensorVirtual.py
FRACOES_DP = np.logspace(-3, 0, num=10)   # same sweep as Leclerc et al., scaled by the standard deviation
N_DA = sorted({5, 6, 8, 10, 13, 17, 22, 29, 38, 50} | {2, 4, 12, 15, 20, 60})
HEARTBEAT = 30

VALIDAR_V11 = True
PARAR_SE_FALHAR = True
TOL_ABS, TOL_REL = 1e-6, 1e-4     # the JSON is rounded to 6 decimal places

# (label in the JSON, device in the paper, type, position), as in sensorVirtual.py
DISPOSITIVOS = [
    ("SC01 (h1)", "Sensor-A", "Temperature", 0),
    ("SC02 (h2)", "Sensor-B", "Temperature", 500),
    ("SC03 (h3)", "Sensor-C", "Temperature", 1000),
    ("SC04 (h4)", "Sensor-D", "Humidity", 0),
    ("SC05 (h5)", "Sensor-E", "Humidity", 500),
    ("SC06 (h6)", "Sensor-F", "Brightness", 0),
    ("SC07 (h7)", "Sensor-G", "Brightness", 500),
    ("SC08 (h8)", "Sensor-H", "Voltage", 0),
]

# POLICY_SCENARIOS of run_mqtt_pipeline_v11.py
POLITICAS = {
    "Baseline": {
        "Temperature": (2, "703-Temp_A2_O2000_J10_E2000_Ep1000_13.h5"),
        "Humidity": (2, "2634-Umi_A2_O2000_J10_E1000_Ep1500_4.h5"),
        "Brightness": (2, "2620-Lum_A2_O2000_J10_E1000_Ep500_10.h5"),
        "Voltage": (2, "2602-Vol_A2_O2000_J10_E1000_Ep100_2.h5"),
    },
    "Conservative": {
        "Temperature": (4, "5416-Temp_A4_O2000_J10_E1000_Ep1000_1.h5"),
        "Humidity": (10, "3620-Umi_A10_O2000_J10_E1000_Ep500_10.h5"),
        "Brightness": (6, "3420-Lum_A6_O2000_J10_E1000_Ep500_10.h5"),
        "Voltage": (60, "4624-Vol_A60_O2000_J10_E1000_Ep1000_4.h5"),
    },
    "Aggressive": {
        "Temperature": (15, "16834-Temp_A15_O2000_J10_E1000_Ep1500_7.h5"),
        "Humidity": (12, "3825-Umi_A12_O2000_J10_E1000_Ep1000_5.h5"),
        "Brightness": (10, "3630-Lum_A10_O2000_J10_E1000_Ep1000_10.h5"),
        "Voltage": (20, "4221-Vol_A20_O2000_J10_E1000_Ep1000_1.h5"),
    },
}

# ----------------------------------------------------------------------------
# Data (same reading logic as sensorVirtual.py)
# ----------------------------------------------------------------------------
def carregar_colunas(caminho, max_linhas):
    cols = {"Temperature": [], "Humidity": [], "Brightness": [], "Voltage": []}
    contagem = 0
    with open(caminho) as f:
        for linha in f:
            if contagem >= max_linhas:
                break
            partes = linha.strip().split()
            if len(partes) >= 8:
                try:
                    cols["Temperature"].append(float(partes[4]))
                    cols["Humidity"].append(float(partes[5]))
                    cols["Brightness"].append(float(partes[6]))
                    cols["Voltage"].append(float(partes[7]))
                    contagem += 1
                except (ValueError, IndexError):
                    continue
    return {k: np.array(v) for k, v in cols.items()}

# ----------------------------------------------------------------------------
# Pipeline functions (same logic as run_mqtt_pipeline_v11.py)
# ----------------------------------------------------------------------------
def aggregate_data(raw_data, ar):
    means = []
    for i in range(0, len(raw_data), ar):
        block = raw_data[i:i + ar]
        means.append(sum(block) / len(block))
    return means


def expand_aggregated(means, ar, n):
    series = [0.0] * n
    for j, mean_val in enumerate(means):
        pos = (j + 1) * ar - 1
        if pos < n:
            series[pos] = mean_val
    return series


def reconstruct_data(aggregated_means, raw_data, model, window_size, ar):
    n = len(raw_data)
    series = expand_aggregated(aggregated_means, ar, n)
    for k in range(min(window_size, n)):
        series[k] = raw_data[k]
    for idx in range(window_size, n):
        if series[idx] == 0:
            window = np.array(series[idx - window_size:idx])[np.newaxis]
            prediction = model.predict(window, verbose=0)
            series[idx] = prediction[0][0]
    return series


def load_perceptron(model_path):
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[PERCEPTRON_J])])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    model.load_weights(model_path)
    return model

# ----------------------------------------------------------------------------
# M2 and metrics
# ----------------------------------------------------------------------------
def recuperar_linear_pipeline(x, ar, J=PERCEPTRON_J):
    """Same values transmitted as M1; each mean placed at the center of its block
        (as in DA); exact seed values in the first J positions.
    """
    n = len(x)
    medias = aggregate_data(list(x), ar)
    centros = [(ini + min(ini + ar, n) - 1) / 2 for ini in range(0, n, ar)]
    pts_t = list(range(J)) + [c for c in centros if c > J - 1]
    pts_v = list(x[:J]) + [m for c, m in zip(centros, medias) if c > J - 1]
    rec = np.interp(np.arange(n), pts_t, pts_v)
    rec[:J] = x[:J]
    return len(medias) + J, rec


def metricas(x, rec):
    x = np.asarray(x, dtype=float)
    d = x - np.asarray(rec, dtype=float)
    mse = float(np.mean(d ** 2))
    rmse = float(np.sqrt(mse))
    faixa = float(x.max() - x.min())
    return {"mae": float(np.mean(np.abs(d))), "mse": mse, "rmse": rmse,
            "nrmse_pct": rmse / faixa * 100 if faixa > 0 else float("nan"),
            "md": float(np.max(np.abs(d)))}


def reducao(n_transmitidos, n=N_OBS):
    return (1 - n_transmitidos / n) * 100

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
CAMPOS = ["metodo", "parametro", "valor", "cenario", "rotulo", "dispositivo", "tipo",
          "n_transmitidos", "reducao_pct", "reducao_eq4_pct", "mae", "mse", "rmse",
          "nrmse_pct", "md"]


def linha(metodo, parametro, valor, cenario, disp, n_tx, met, reducao_eq4=""):
    rotulo, nome, tipo, _ = disp
    return {"metodo": metodo, "parametro": parametro, "valor": valor, "cenario": cenario,
            "rotulo": rotulo, "dispositivo": nome, "tipo": tipo, "n_transmitidos": n_tx,
            "reducao_pct": reducao(n_tx), "reducao_eq4_pct": reducao_eq4, **met}


def executar_m1(segmentos, registros):
    """Perceptron reconstruction; reuses cached reconstructions when available."""
    pasta_cache = os.path.join(SAIDA, "reconstrucoes_m1")
    os.makedirs(pasta_cache, exist_ok=True)
    modelos = {}
    for cenario, pol in POLITICAS.items():
        for disp in DISPOSITIVOS:
            rotulo, nome, tipo, _ = disp
            ar, arquivo = pol[tipo]
            x = segmentos[rotulo]
            cache = os.path.join(pasta_cache, f"{cenario}_{nome}.npy")
            if os.path.exists(cache):
                rec = np.load(cache)
            else:
                if arquivo not in modelos:
                    modelos[arquivo] = load_perceptron(os.path.join(PASTA_MODELOS, arquivo))
                t0 = time.time()
                rec = np.array(reconstruct_data(aggregate_data(list(x), ar), list(x),
                                                modelos[arquivo], PERCEPTRON_J, ar))
                np.save(cache, rec)
                print(f"  M1 {cenario:12s} {nome}: {time.time() - t0:.1f}s")
            n_medias = len(aggregate_data(list(x), ar))
            registros.append(linha("M1_pipeline", "cenario", cenario, cenario, disp,
                                   n_medias + PERCEPTRON_J, metricas(x, rec),
                                   reducao(n_medias)))


def validar_v11(registros):
    with open(JSON_V11) as f:
        ref = json.load(f)["scenarios"]
    saida, falhas = [], 0
    for r in registros:
        if r["metodo"] != "M1_pipeline":
            continue
        esperado = ref[r["cenario"]][r["rotulo"]]
        for chave in ("mae", "mse"):
            obtido, alvo = r[chave], esperado[chave]
            ok = abs(obtido - alvo) <= max(TOL_ABS, TOL_REL * abs(alvo))
            falhas += not ok
            saida.append({"cenario": r["cenario"], "dispositivo": r["dispositivo"],
                          "metrica": chave, "json_v11": alvo, "obtido": obtido,
                          "diferenca": obtido - alvo, "ok": ok})
    with open(os.path.join(SAIDA, "validacao_v11.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(saida[0].keys()))
        w.writeheader()
        w.writerows(saida)
    print(f"Validacao contra o v11: {len(saida) - falhas}/{len(saida)} valores reproduzidos")
    return falhas == 0


def executar_linhas_de_base(segmentos, desvios, registros):
    for disp in DISPOSITIVOS:
        rotulo, nome, tipo, _ = disp
        x = segmentos[rotulo]
        for cenario, pol in POLITICAS.items():
            n_tx, rec = recuperar_linear_pipeline(x, pol[tipo][0])
            registros.append(linha("M2_pipeline_linear", "cenario", cenario, cenario, disp,
                                   n_tx, metricas(x, rec)))
        for n in N_DA:
            n_tx, rec = da_leclerc(x, n)
            registros.append(linha("M3_DA", "N", n, "", disp, n_tx, metricas(x, rec)))
        for frac in FRACOES_DP:
            n_tx, rec = df_leclerc(x, frac * desvios[tipo], max_interval=HEARTBEAT)
            registros.append(linha("M4_DF", "fracao_desvio", frac, "", disp, n_tx, metricas(x, rec)))
            n_tx, rec = dp_leclerc(x, frac * desvios[tipo], max_interval=HEARTBEAT)
            registros.append(linha("M5_DP", "fracao_desvio", frac, "", disp, n_tx, metricas(x, rec)))


def resumir(registros):
    """Mean over the 8 devices for each method and parameter."""
    grupos = {}
    for r in registros:
        grupos.setdefault((r["metodo"], str(r["valor"])), []).append(r)
    resumo = []
    for (metodo, valor), rs in grupos.items():
        resumo.append({"metodo": metodo, "valor": valor, "n_dispositivos": len(rs),
                       **{k: float(np.mean([r[k] for r in rs]))
                          for k in ("reducao_pct", "mae", "nrmse_pct", "md")}})
    return resumo


def curva(registros, metodo, rotulo):
    rs = [r for r in registros if r["metodo"] == metodo and r["rotulo"] == rotulo]
    red = np.array([r["reducao_pct"] for r in rs])
    unicos = np.unique(red)
    return (unicos,
            np.array([np.mean([r["nrmse_pct"] for r, v in zip(rs, red) if v == u]) for u in unicos]),
            np.array([np.mean([r["mae"] for r, v in zip(rs, red) if v == u]) for u in unicos]))


def isotrafego(registros):
    """Error of each baseline interpolated, per device, at the M1 reduction."""
    saida = []
    for cenario in POLITICAS:
        m1 = {r["rotulo"]: r for r in registros
              if r["metodo"] == "M1_pipeline" and r["cenario"] == cenario}
        m2 = {r["rotulo"]: r for r in registros
              if r["metodo"] == "M2_pipeline_linear" and r["cenario"] == cenario}
        linha_saida = {"cenario": cenario,
                       "reducao_media_pct": np.mean([r["reducao_pct"] for r in m1.values()]),
                       "M1_nrmse": np.mean([r["nrmse_pct"] for r in m1.values()]),
                       "M1_mae": np.mean([r["mae"] for r in m1.values()]),
                       "M2_nrmse": np.mean([r["nrmse_pct"] for r in m2.values()]),
                       "M2_mae": np.mean([r["mae"] for r in m2.values()])}
        for metodo in ("M3_DA", "M4_DF", "M5_DP"):
            nr, ma, validos = [], [], []
            for rotulo, r1 in m1.items():
                red, nrmse, mae = curva(registros, metodo, rotulo)
                alvo = r1["reducao_pct"]
                if len(red) >= 2 and red.min() <= alvo <= red.max():
                    nr.append(np.interp(alvo, red, nrmse))
                    ma.append(np.interp(alvo, red, mae))
                    validos.append(rotulo)
            # M1 is compared on the same subset of devices for which the baseline
            # reaches the M1 reduction (no extrapolation outside the curve)
            linha_saida[f"{metodo}_n_disp"] = len(validos)
            linha_saida[f"{metodo}_nrmse"] = np.mean(nr) if nr else float("nan")
            linha_saida[f"{metodo}_mae"] = np.mean(ma) if ma else float("nan")
            linha_saida[f"{metodo}_M1_nrmse_mesmos_disp"] = (
                np.mean([m1[r]["nrmse_pct"] for r in validos]) if validos else float("nan"))
            linha_saida[f"{metodo}_M1_mae_mesmos_disp"] = (
                np.mean([m1[r]["mae"] for r in validos]) if validos else float("nan"))
        saida.append(linha_saida)
    return saida


def gravar_csv(nome, linhas, campos=None):
    campos = campos or list(linhas[0].keys())
    with open(os.path.join(SAIDA, nome), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def figuras(resumo, registros):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estilos = {"M3_DA": ("DA (Leclerc et al.)", "s", "tab:orange"),
               "M4_DF": ("DF (Leclerc et al.)", "D", "tab:green"),
               "M5_DP": ("DP (Leclerc et al.)", "o", "tab:blue")}

    def desenhar(ax, regs, titulo):
        for metodo, (nome, marca, cor) in estilos.items():
            pts = sorted((np.mean([r["reducao_pct"] for r in regs if r["metodo"] == metodo and str(r["valor"]) == v]),
                          np.mean([r["nrmse_pct"] for r in regs if r["metodo"] == metodo and str(r["valor"]) == v]))
                         for v in {str(r["valor"]) for r in regs if r["metodo"] == metodo})
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marca, color=cor,
                    label=nome, markersize=4, linewidth=1)
        for metodo, nome, cheio in (("M1_pipeline", "Pipeline (perceptron)", True),
                                    ("M2_pipeline_linear", "Pipeline (linear)", False)):
            for cenario, marca in zip(POLITICAS, ("^", "*", "P")):
                rs = [r for r in regs if r["metodo"] == metodo and r["cenario"] == cenario]
                ax.plot(np.mean([r["reducao_pct"] for r in rs]), np.mean([r["nrmse_pct"] for r in rs]),
                        marker=marca, linestyle="none", markersize=9, color="black",
                        markerfacecolor="black" if cheio else "none",
                        label=f"{nome}, {cenario}")
        ax.set_xlabel("Traffic reduction (%)")
        ax.set_ylabel("NRMSE (%)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(titulo)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    desenhar(ax, registros, "Mean over the eight devices")
    ax.legend(fontsize=7, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SAIDA, f"fig_comparacao_nrmse.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, eixos = plt.subplots(2, 2, figsize=(11, 8))
    for ax, tipo in zip(eixos.flat, ("Temperature", "Humidity", "Brightness", "Voltage")):
        desenhar(ax, [r for r in registros if r["tipo"] == tipo], tipo)
    eixos.flat[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SAIDA, f"fig_comparacao_por_tipo.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(SAIDA, exist_ok=True)
    cols = carregar_colunas(ARQUIVO_DADOS, MAX_LINHAS)
    desvios = {tipo: float(np.std(v)) for tipo, v in cols.items()}
    segmentos = {}
    for rotulo, nome, tipo, pos in DISPOSITIVOS:
        seg = cols[tipo][pos:pos + N_OBS]
        if len(seg) < N_OBS:
            sys.exit(f"Segmento curto para {nome} ({tipo}, posicao {pos}).")
        segmentos[rotulo] = seg

    with open(os.path.join(SAIDA, "metadados.json"), "w") as f:
        json.dump({"arquivo_dados": ARQUIVO_DADOS, "linhas_validas": {k: len(v) for k, v in cols.items()},
                   "desvio_padrao_por_tipo": desvios, "fracoes_df_dp": list(FRACOES_DP),
                   "n_da": N_DA, "heartbeat": HEARTBEAT, "perceptron_J": PERCEPTRON_J,
                   "reducao": "Equacao 4 contando todos os valores transmitidos; "
                              "na pipeline, medias + J seeds (reducao_eq4_pct conta so as medias)"},
                  f, indent=2)

    registros = []
    print("M1: reconstrucao por perceptron")
    executar_m1(segmentos, registros)
    if VALIDAR_V11 and not validar_v11(registros) and PARAR_SE_FALHAR:
        sys.exit("M1 nao reproduz o v11. Ver validacao_v11.csv antes de comparar.")

    print("M2 a M5")
    executar_linhas_de_base(segmentos, desvios, registros)

    gravar_csv("resultados_por_dispositivo.csv", registros, CAMPOS)
    gravar_csv("resumo_por_configuracao.csv", resumir(registros))
    gravar_csv("isotrafego.csv", isotrafego(registros))
    figuras(resumir(registros), registros)
    print(f"Concluido. Resultados em {SAIDA}/")


if __name__ == "__main__":
    main()
