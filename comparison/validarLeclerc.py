"""validarLeclerc.py

Checks that leclerc_metodos.py reproduces the results published in the repository
of Leclerc et al. (2026). Run inside a clone of
https://github.com/SebastianLeclerc/edge-data-reduction, with leclerc_metodos.py
on the PYTHONPATH.

For each scenario (steady, rising, fluctuating) and each parameter of the sweep,
compares the number of transmitted points, TAD and MD with the files
*_log_scale_results.txt. Uses replicar_defeito_ponta=True (DF and DA) and time
arithmetic in microseconds (DP) only to reproduce the original code exactly.
"""

import datetime
import numpy as np
from leclerc_metodos import df_leclerc, da_leclerc, dp_leclerc

CENARIOS = {"Stable": "steady", "Rising": "rising", "Fluctuating": "fluctuating"}
LIMIARES = np.logspace(-3, 0, num=10)       # exact values used in the original


def carregar_baseline(cenario):
    t, v = [], []
    with open(f"baseline_{cenario}.csv") as f:
        for linha in f:
            a, b = linha.strip().split(",")
            t.append(datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S.%f"))
            v.append(float(b))
    us = np.array([(x - t[0]) // datetime.timedelta(microseconds=1) for x in t], dtype=float)
    return us, np.array(v)


def carregar_publicados(arquivo):
    """Read the result tables; the header label is used only for the scenario."""
    saida, atual = {}, None
    with open(arquivo) as f:
        for linha in f:
            linha = linha.strip()
            if linha.startswith("#"):
                atual = CENARIOS[linha.split()[2]]
                saida[atual] = []
                continue
            partes = [p.strip() for p in linha.split("|")]
            if len(partes) == 6:
                try:
                    saida[atual].append([float(p) for p in partes])
                except ValueError:
                    pass
    return saida


def main():
    publicados = {m: carregar_publicados(f"{m}_log_scale_results.txt") for m in ("DF", "DA", "DP")}
    total = {m: [0, 0] for m in publicados}
    for cenario in CENARIOS.values():
        t_us, v = carregar_baseline(cenario)
        t_s = t_us / 1e6
        for metodo, tabelas in publicados.items():
            for linha in tabelas[cenario]:
                p, enviados_pub, _, tad_pub, _, md_pub = linha
                if metodo == "DF":
                    p = float(min(LIMIARES, key=lambda z: abs(z - p)))
                    k, rec = df_leclerc(v, p, t_s, replicar_defeito_ponta=True)
                elif metodo == "DA":
                    k, rec = da_leclerc(v, int(p), t_s, replicar_defeito_ponta=True)
                else:
                    p = float(min(LIMIARES, key=lambda z: abs(z - p)))
                    k, rec = dp_leclerc(v, p, t_us, escala_tempo=1e6)
                d = np.abs(v - rec)
                tad, md = d.sum(), d.max()
                ok = (k == int(enviados_pub)
                      and abs(tad - tad_pub) <= 0.006 + 1e-3 * tad_pub
                      and abs(md - md_pub) <= 0.006)
                total[metodo][0] += ok
                total[metodo][1] += 1
                if not ok:
                    print(f"DIVERGE {metodo} {cenario} p={p:.4f}: publicado "
                          f"{int(enviados_pub)}/{tad_pub}/{md_pub}, obtido {k}/{tad:.4f}/{md:.4f}")
    for metodo, (ok, n) in total.items():
        print(f"{metodo}: {ok}/{n} configuracoes reproduzidas")


if __name__ == "__main__":
    main()
