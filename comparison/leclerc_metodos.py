"""leclerc_metodos.py

Reimplementation of the three edge data reduction methods evaluated in
  S. Leclerc, E. Hansen, A. Bucaioni, M. Ashjaei, "Reducing IoT Data at the Edge:
  A Comparative Evaluation", IEEE ICIT 2026, doi:10.1109/ICIT64854.2026.11491260.
Original code: https://github.com/SebastianLeclerc/edge-data-reduction
(the repository has no license, so the logic is reimplemented here rather than copied).

Correspondence with the original code (DF_log.py, DA_log.py, DP_log.py):
  DF  fixed threshold delta on the last transmitted value; heartbeat that doubles
      from 1 to 30 points without transmission and returns to 1 when the
      threshold is exceeded.
  DA  first point sent on its own (it is also part of the first block); mean of
      each block of N points placed at the mean time of the block; the last,
      incomplete block is also sent.
  DP  linear prediction from the last anchor point; when the margin eps is
      violated, sends the previous and the current points and recomputes the
      slope; heartbeat every 30 counts (the counter is incremented twice per
      point within the margin, as in the original, which results in one
      transmission every ~15 points).
  Recovery: linear interpolation over time between transmitted points.

Single intentional difference: in the original DF and DA, the points after the
last transmitted point receive the FIRST transmitted value instead of the last
one (DF_log.py, lines 60-61 and 79-80; DA_log.py, lines 64-70, where the branch
"if not before" also captures the instants after the last point and the branch
"if not after" is never reached). Here these points receive the LAST transmitted
value, as in DP. replicar_defeito_ponta=True reproduces the original behavior
and is used only to validate this reimplementation against the published results.
"""

import numpy as np


def _recuperar(t, pts_t, pts_v):
    """Linear interpolation; outside the range, the value of the nearest end."""
    return np.interp(t, np.asarray(pts_t, dtype=float), np.asarray(pts_v, dtype=float))


def df_leclerc(x, delta, t=None, min_interval=1, max_interval=30,
               replicar_defeito_ponta=False):
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
    enviados = []
    ultimo = None
    contador = 0
    intervalo = min_interval
    for i, v in enumerate(x):
        enviar = False
        if ultimo is None:
            enviar = True
        elif abs(v - ultimo) >= delta:
            enviar = True
            intervalo = min_interval
        elif contador >= intervalo:
            enviar = True
            intervalo = min(max_interval, intervalo * 2)
        if enviar:
            enviados.append(i)
            ultimo = v
            contador = 0
        else:
            contador += 1
    pts_t, pts_v = t[enviados], x[enviados]
    rec = _recuperar(t, pts_t, pts_v)
    if replicar_defeito_ponta:
        rec[t > pts_t[-1]] = pts_v[0]
    return len(enviados), rec


def da_leclerc(x, n, t=None, replicar_defeito_ponta=False):
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
    pts_t, pts_v = [t[0]], [x[0]]            # first point sent on its own
    for ini in range(0, len(x), n):          # blocks of n points (the last one may be incomplete)
        bloco = slice(ini, min(ini + n, len(x)))
        pts_t.append(t[bloco].mean())
        pts_v.append(x[bloco].mean())
    rec = _recuperar(t, pts_t, pts_v)
    if replicar_defeito_ponta:
        rec[t >= pts_t[-1]] = pts_v[0]
    return len(pts_v), rec


def dp_leclerc(x, eps, t=None, max_interval=30, escala_tempo=1.0):
    """escala_tempo is used only for validation: with t in integer microseconds and
        escala_tempo=1e6, time differences are computed exactly as the
        timedelta.total_seconds() of the original code.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, dtype=float)
    salvos = []                               # indices, without consecutive repetition

    def salvar(i):
        if not salvos or salvos[-1] != i:
            salvos.append(i)

    ultimo_i = None
    ancora_i = None
    coef = 0.0
    contador = 0
    for i, v in enumerate(x):
        enviar = False
        recalculado = False
        if contador >= max_interval:
            enviar = True
            contador = 0
        elif ultimo_i is None:
            enviar = True
            recalculado = True
        else:
            previsto = x[ancora_i] + coef * ((t[i] - t[ancora_i]) / escala_tempo)
            if abs(previsto - v) < eps:
                contador += 1
            else:
                coef = (v - x[ultimo_i]) / ((t[i] - t[ultimo_i]) / escala_tempo)
                salvar(ultimo_i)
                enviar = True
                recalculado = True
        ultimo_i = i
        if enviar:
            salvar(i)
            if recalculado:
                ancora_i = i
            contador = 0
        else:
            contador += 1
    rec = _recuperar(t, t[salvos], x[salvos])
    return len(salvos), rec
