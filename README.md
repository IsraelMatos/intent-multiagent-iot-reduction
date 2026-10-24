# Intent-driven multi-agent orchestration for sensor-aware IoT data reduction

Code, trained models, and result files for the paper

> I. N. Matos, E. P. Batista Jr., C. Jr. N. da Silva, G. P. Pinto, F. A. Durao, M. L. M. Peixoto, G. B. Figueiredo, C. V. S. Prazeres.
> *Toward Agentic Edge AI: Intent-Driven Multi-Agent Orchestration for Sensor-Aware IoT Data Reduction.* Computer Communications (under review).

Six SPADE agents (Classifier, Orchestrator, Registration, Policy, Aggregator, and Reconstructor) coordinate through FIPA-ACL over XMPP, while sensor data flows through two MQTT brokers. Unknown sensors are classified by an LSTM model, the PolicyAgent resolves an aggregation rate and a recovery model as a function of the active intent and the inferred sensor type, data is aggregated at the edge by block means, and the signal is recovered in the cloud.

The FoT-Simulator topology and the virtual devices used in the workload experiments are in a separate repository:
https://github.com/ErnandoPassosJr/fot-sim-exps-agentic-ai

## Repository layout

```
pipeline/
  run_mqtt_pipeline_v11.py      multi-agent pipeline, eight-device functional experiment
  run_mqtt_pipeline_v12.py      same pipeline with concurrency and scalability instrumentation
  sensorVirtual.py              local replacement for the FoT-Simulator publisher (v11 runs)
classifier/
  classificarSensores2.py       LSTM grid (sliding vs. tumbling) and final configuration
  resumirResultados.py          summary of the classifier results (Section 6.1)
  converterClassificador.py     converts the final classifier to the format loaded by the pipeline
comparison/
  leclerc_metodos.py            reimplementation of DF, DA and DP (Leclerc et al., IEEE ICIT 2026)
  validarLeclerc.py             validation of the reimplementation against the published results
  compararReducao.py            comparison at equal traffic (Table 10)
  analiseConscienciaTipo.py     per-type vs. uniform parameters at equal traffic
  analiseRequisitoQualidade.py  traffic under a common per-sensor quality requirement (Table 11)
models/                         trained models (see models/README.md)
results/
  pipeline/                     JSON output of the pipeline runs
  lstm/                         runs.csv and metadados.json of the classifier grid and final runs
  comparison/                   CSV output of the comparison scripts
```

## Data

All experiments use the Intel Berkeley Research Lab dataset, available at
https://db.csail.mit.edu/labdata/labdata.html. Download `data.txt.gz`, extract `data.txt`, and copy it (or create a link to it) in the folder from which each script is run. The scripts read the first lines of the file, which correspond to mote 1, and use fields 4 to 7 (temperature, humidity, light, and voltage). The file is not redistributed here.

## Environments

| Folder | Python | Main packages |
|---|---|---|
| `pipeline/` | 3.11 | tensorflow 2.15.0, spade 3.3.2, paho-mqtt 2.x, numpy, psutil (optional) |
| `classifier/` | 3.9 | tensorflow 2.20.0 (grid) and 2.17.0 (final runs), scikit-learn, scipy, numpy, matplotlib |
| `comparison/` | 3.11 | same environment as the pipeline (tensorflow is needed only for the perceptron recovery in `compararReducao.py`), matplotlib |

The requirement files are in `requirements/`.

The pipeline also needs:
- an XMPP server on `localhost:5222` (Prosody was used) with in-band registration enabled, since the agents register their accounts on startup (`classifier@localhost`, `orchestrator@localhost`, `registration@localhost`, `policy@localhost`, `aggregator@localhost`, `reconstructor@localhost`);
- two MQTT brokers accepting anonymous connections: the edge broker on `localhost:1883` and the cloud broker on `localhost:1884` (Mosquitto was used).

## Reproducing the results

Section and table numbers refer to the paper.

**Classifier (Section 6.1, Tables 4 and 5).** In `classifier/`, with `data.txt`:

1. Run `python classificarSensores2.py` with `MODO = "grid"`. Results go to `Resultados2_grid/`.
2. Set `MODO = "final"` and `NORMALIZACOES = ["nenhuma", "zscore_janela"]` and run it again. The configuration is selected from the grid using the validation accuracy only and repeated with five seeds. Results go to `Resultados2_final/`.
3. Run `python resumirResultados.py` to obtain the statistics reported in Section 6.1.

The files produced by these steps are in `results/lstm/`. In the final run, the fields `observacoes` and `janelas` of `metadados.json` list the grid values; the configuration actually used is the one in `runs.csv`.

To use the final model in the pipeline, run `python converterClassificador.py exportar` in the classifier environment and `python converterClassificador.py gerar` in the pipeline environment (see the header of the script). This writes `model_classifier.h5`.

**Functional experiment (Sections 6.2 to 6.5, Tables 6 to 9).** In `pipeline/`, with `data.txt`, `model_classifier.h5` and the 12 perceptron models:

1. Start the XMPP server and the two MQTT brokers.
2. `python sensorVirtual.py --dataset data.txt --sensors 8`
3. `python run_mqtt_pipeline_v11.py`

The output is written to `mqtt_pipeline_results_v11.json` (see `results/pipeline/`).

**Workload and scalability (Section 6.7).** The devices are created by the FoT-Simulator (repository above), which publishes the sensor list and the data streams on the edge broker. The pipeline is run with

```
python run_mqtt_pipeline_v12.py --concurrency N --fixed-types
```

where `--concurrency 1` gives the serial mode and `--concurrency N` the concurrent mode. With `--fixed-types`, the LSTM still runs and its latency is measured, but the policy uses the sensor type reported by the simulator, so every run processes the same workload.

**Comparison with edge data reduction methods (Section 6.6, Tables 10 and 11).** In `comparison/`, with `data.txt`, the 12 perceptron models and `mqtt_pipeline_results_v11.json`:

1. `python compararReducao.py` first checks that the perceptron recovery reproduces the MAE and MSE of the v11 run, then writes `Resultados_comparacao/isotrafego.csv` and the other comparison files.
2. `python analiseConscienciaTipo.py`
3. `python analiseRequisitoQualidade.py`

`validarLeclerc.py` checks the reimplementation of DF, DA and DP. It must be run inside a clone of https://github.com/SebastianLeclerc/edge-data-reduction, with `leclerc_metodos.py` on the `PYTHONPATH`.

## Perceptron models

The 12 perceptron recovery models were trained in previous work and are used here without retraining:

> I. N. Matos, G. B. Figueiredo, M. L. M. Peixoto, P. K. Donta, S. Dustdar, C. Prazeres. Neural networks-based accurate IoT data reconstruction. *Telematics and Informatics Reports* 22, 100315, 2026.
