"""analiseRequisitoQualidade.py

Question: for all sensors to meet the same quality requirement, how much traffic
must be transmitted when the reduction parameter is chosen per sensor type,
compared with the best single parameter chosen without knowing the type?

Protocol:
  Methods      AGG: block mean + linear interpolation (M2 recovery, means + 10 seed values)
               DF and DP: Leclerc et al. (2026), heartbeat 30
               The perceptron is not included: models exist for only 3 values of ar per type.
  Grids        AGG: ar in GRADE_AR. DF and DP: absolute threshold in GRADE_LIMIAR (same
               grid for both strategies; the only difference is choosing per type).
  Requirement  segment MSE <= k x MSE_ref(type), where MSE_ref(type) is the mean MSE of
               the calibration segments of the type under AGG with ar=2.
               k = 3 is the main criterion (the factor of the Conservative criterion);
               k = 2, 5 and 10 are sensitivity levels.
  Calibration  10 segments per type, N=1000, positions 3000 + 1800 i (i = 0..9),
               disjoint from the evaluation segments (lines 0 to 2000).
  Selection    A parameter is admissible if ALL calibration segments considered meet
               the requirement. Per type: segments of the type. Single: segments of all
               types. Among the admissible values, the one with the highest mean
               calibration reduction is chosen. If none is admissible, the least
               aggressive value is used and the choice is marked as infeasible.
  Evaluation   The 8 segments of the functional experiment (not used for calibration):
               mean reduction, transmitted traffic and number of sensors that meet the
               requirement.
  Validity     A segment is used only if its timestamps (date and time) are strictly
               increasing, i.e. if it is a continuous time series. In data.txt, the
               epoch counter of mote 1 repeats after line ~19000 and readings from
               different days appear interleaved. Invalid calibration segments are
               discarded; if an evaluation segment is invalid, the script stops.

Run in the same folder as compararReducao.py. Does not use TensorFlow; takes about 1 minute.
"""

import csv
import json
import os

import numpy as np

from compararReducao import (ARQUIVO_DADOS, DISPOSITIVOS, HEARTBEAT, MAX_LINHAS, N_OBS, SAIDA,
                             carregar_colunas, metricas, recuperar_linear_pipeline, reducao)
from leclerc_metodos import df_leclerc, dp_leclerc

K_PRINCIPAL = 3
K_SENSIBILIDADE = [2, 5, 10]
GRADE_AR = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]
GRADE_LIMIAR = list(np.logspace(-5, 3, num=65))      # 8 values per decade
POSICOES_CALIBRACAO = [3000 + 1800 * i for i in range(10)]
TIPOS = ["Temperature", "Humidity", "Brightness", "Voltage"]
METODOS = {"AGG": GRADE_AR, "DF": GRADE_LIMIAR, "DP": GRADE_LIMIAR}


def carregar_tempos(caminho, max_linhas):
    """Same line selection as carregar_colunas (8 fields, fields 4 to 7 numeric),
        keeping date and time in seconds.
    """
    import datetime
    tempos = []
    with open(caminho) as f:
        for linha in f:
            if len(tempos) >= max_linhas:
                break
            p = linha.strip().split()
            if len(p) >= 8:
                try:
                    [float(v) for v in p[4:8]]
                except ValueError:
                    continue
                data = datetime.date.fromisoformat(p[0])
                h, m, seg = p[1].split(":")
                tempos.append(data.toordinal() * 86400 + int(h) * 3600 + int(m) * 60 + float(seg))
    return np.array(tempos)


def segmento_valido(tempos, pos):
    return bool(np.all(np.diff(tempos[pos:pos + N_OBS]) > 0))


def aplicar(metodo, x, p):
    if metodo == "AGG":
        n_tx, rec = recuperar_linear_pipeline(x, int(p))
    elif metodo == "DF":
        n_tx, rec = df_leclerc(x, p, max_interval=HEARTBEAT)
    else:
        n_tx, rec = dp_leclerc(x, p, max_interval=HEARTBEAT)
    return reducao(n_tx), metricas(x, rec)["mse"]


def main():
    cols = carregar_colunas(ARQUIVO_DADOS, MAX_LINHAS)

    # segments: (id, type, series, set)
    segmentos = []
    for tipo in TIPOS:
        for pos in POSICOES_CALIBRACAO:
            segmentos.append((f"cal-{tipo}-{pos}", tipo, cols[tipo][pos:pos + N_OBS], "calibracao"))
    for rotulo, nome, tipo, pos in DISPOSITIVOS:
        segmentos.append((nome, tipo, cols[tipo][pos:pos + N_OBS], "avaliacao"))
    for sid, _, x, _ in segmentos:
        if len(x) < N_OBS:
            raise SystemExit(f"Segmento curto: {sid}")

    tempos = carregar_tempos(ARQUIVO_DADOS, MAX_LINHAS)
    pos_de = {f"cal-{t}-{p}": p for t in TIPOS for p in POSICOES_CALIBRACAO}
    pos_de.update({nome: pos for _, nome, _, pos in DISPOSITIVOS})
    excluidos = [s[0] for s in segmentos if not segmento_valido(tempos, pos_de[s[0]])]
    invalidos_aval = [sid for sid in excluidos if not sid.startswith("cal-")]
    if invalidos_aval:
        raise SystemExit(f"Segmento de avaliacao invalido: {invalidos_aval}")
    segmentos = [s for s in segmentos if s[0] not in excluidos]
    print("Segmentos de calibracao excluidos (tempo nao crescente):", excluidos or "nenhum")

    # resultados[method][(sid, p)] = (reduction, mse)
    resultados = {m: {} for m in METODOS}
    for metodo, grade in METODOS.items():
        for sid, _, x, _ in segmentos:
            for p in grade:
                resultados[metodo][(sid, p)] = aplicar(metodo, x, p)

    calib = {t: [s for s in segmentos if s[3] == "calibracao" and s[1] == t] for t in TIPOS}
    todos_calib = [s for s in segmentos if s[3] == "calibracao"]
    avaliacao = [s for s in segmentos if s[3] == "avaliacao"]
    ref = {t: float(np.mean([resultados["AGG"][(s[0], 2)][1] for s in calib[t]])) for t in TIPOS}

    def escolher(metodo, segs, k):
        grade = METODOS[metodo]
        red_media = {p: np.mean([resultados[metodo][(s[0], p)][0] for s in segs]) for p in grade}
        admiss = [p for p in grade
                  if all(resultados[metodo][(s[0], p)][1] <= k * ref[s[1]] for s in segs)]
        if admiss:
            return max(admiss, key=lambda p: red_media[p]), True
        return min(grade, key=lambda p: red_media[p]), False

    def avaliar(metodo, escolha_por_tipo, k):
        reds, cumpre = [], []
        for sid, tipo, _, _ in avaliacao:
            red, mse = resultados[metodo][(sid, escolha_por_tipo[tipo])]
            reds.append(red)
            cumpre.append(mse <= k * ref[tipo])
        return float(np.mean(reds)), sum(cumpre), cumpre

    linhas_escolha, linhas_aval = [], []
    for k in [K_PRINCIPAL] + K_SENSIBILIDADE:
        for metodo in METODOS:
            por_tipo = {t: escolher(metodo, calib[t], k) for t in TIPOS}
            unico, viavel_u = escolher(metodo, todos_calib, k)
            estrategias = {"por_tipo": {t: v[0] for t, v in por_tipo.items()},
                           "uniforme": {t: unico for t in TIPOS}}
            viabilidade = {"por_tipo": all(v[1] for v in por_tipo.values()), "uniforme": viavel_u}
            for nome, escolha in estrategias.items():
                red, n_ok, cumpre = avaliar(metodo, escolha, k)
                linhas_aval.append({
                    "k": k, "metodo": metodo, "estrategia": nome,
                    "viavel_na_calibracao": viabilidade[nome],
                    "reducao_media_pct": red, "trafego_transmitido_pct": 100 - red,
                    "sensores_que_cumprem": n_ok, "total_sensores": len(avaliacao),
                    **{f"cumpre_{s[0]}": c for s, c in zip(avaliacao, cumpre)}})
                for t in TIPOS:
                    linhas_escolha.append({"k": k, "metodo": metodo, "estrategia": nome, "tipo": t,
                                           "parametro": escolha[t],
                                           "viavel": por_tipo[t][1] if nome == "por_tipo" else viavel_u,
                                           "no_limite_da_grade": escolha[t] == max(METODOS[metodo])})

    os.makedirs(SAIDA, exist_ok=True)
    for nome, dados in (("requisito_escolhas.csv", linhas_escolha),
                        ("requisito_avaliacao.csv", linhas_aval)):
        with open(os.path.join(SAIDA, nome), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dados[0].keys()))
            w.writeheader()
            w.writerows(dados)
    with open(os.path.join(SAIDA, "requisito_protocolo.json"), "w") as f:
        json.dump({"k_principal": K_PRINCIPAL, "k_sensibilidade": K_SENSIBILIDADE,
                   "grade_ar": GRADE_AR, "grade_limiar": [float(v) for v in GRADE_LIMIAR],
                   "posicoes_calibracao": POSICOES_CALIBRACAO, "mse_ref_por_tipo": ref,
                   "heartbeat": HEARTBEAT, "N": N_OBS,
                   "criterio_validade": "carimbos de tempo estritamente crescentes no segmento",
                   "segmentos_excluidos": excluidos}, f, indent=2)

    print("MSE de referencia por tipo (AGG, ar=2, calibracao):",
          {t: f"{v:.3g}" for t, v in ref.items()})
    for k in [K_PRINCIPAL] + K_SENSIBILIDADE:
        print(f"\nk = {k}{'  (criterio principal)' if k == K_PRINCIPAL else ''}")
        for metodo in METODOS:
            linhas = {l["estrategia"]: l for l in linhas_aval if l["k"] == k and l["metodo"] == metodo}
            pt, un = linhas["por_tipo"], linhas["uniforme"]
            razao = un["trafego_transmitido_pct"] / pt["trafego_transmitido_pct"]
            esc = {l["tipo"][0]: l["parametro"] for l in linhas_escolha
                   if l["k"] == k and l["metodo"] == metodo and l["estrategia"] == "por_tipo"}
            topo = max(METODOS[metodo])
            esc_txt = ", ".join(f"{t}={v:g}{'*' if v == topo else ''}" for t, v in esc.items())
            esc_u = next(l["parametro"] for l in linhas_escolha if l["k"] == k and l["metodo"] == metodo
                         and l["estrategia"] == "uniforme")
            print(f"  {metodo}: por tipo transmite {pt['trafego_transmitido_pct']:5.1f}% "
                  f"(cumprem {pt['sensores_que_cumprem']}/8{'' if pt['viavel_na_calibracao'] else ', INVIAVEL'}) "
                  f"| uniforme {un['trafego_transmitido_pct']:5.1f}% "
                  f"(cumprem {un['sensores_que_cumprem']}/8{'' if un['viavel_na_calibracao'] else ', INVIAVEL'}) "
                  f"| razao {razao:.2f}x | por tipo: {esc_txt} | uniforme: {esc_u:g}{'*' if esc_u == topo else ''}")
    print("\n* parametro no limite superior da grade: a escolha pode estar limitada pela grade.")


if __name__ == "__main__":
    main()
