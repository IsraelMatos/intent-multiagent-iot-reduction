"""Intent-driven multi-agent IoT pipeline over MQTT (v12).

Same agents, policies, aggregation, and reconstruction as v11, with the
instrumentation used in the workload and scalability experiments:

- concurrent operation of up to C devices (--concurrency), limited by a semaphore;
- reconstruction offloaded to a thread pool (--recon-workers), with one lock per model;
- optional fixed-type mode (--fixed-types): the LSTM still runs and its latency is
  measured, but the policy uses the true sensor type reported by the FoT-Simulator,
  so every run processes an identical workload;
- makespan, end-to-end latency, reconstruction time, broker ingress rate, CPU and
  memory sampling.

Sensor discovery and operation data follow v11: the sensor list is read from
iot/edge/config/hosts/<key>, and operation data from iot/edge/raw/{type}/{device_id}.

Usage:
    python run_mqtt_pipeline_v12.py --concurrency 8 --fixed-types
"""

import ast
import argparse
import asyncio
import json
import time
import threading
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import psutil  # optional: CPU/memory sampling
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PROC = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import paho.mqtt.client as mqtt
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

# ============================================================
#  CONFIGURATION
# ============================================================

EDGE_BROKER_HOST = "localhost"
EDGE_BROKER_PORT = 1883
CLOUD_BROKER_HOST = "localhost"
CLOUD_BROKER_PORT = 1884

NUM_OBSERVATIONS = 1000
PERCEPTRON_J = 10
WINDOW_SIZE = 3
SENSOR_TYPES = ["Temperature", "Humidity", "Brightness", "Voltage"]
SENSOR_LABELS = {0: "Temperature", 1: "Humidity", 2: "Brightness", 3: "Voltage"}

# Mapping from the FoT-Simulator sensorType to the pipeline sensor_type
SENSOR_TYPE_MAP = {
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "light":       "Brightness",
    "voltage":     "Voltage",
}

POLICY_SCENARIOS = {
    "Baseline": {
        "description": "Fixed ar=2 for all sensors (no differentiation)",
        "Temperature": {"ar": 2,  "model": "703-Temp_A2_O2000_J10_E2000_Ep1000_13.h5"},
        "Humidity":    {"ar": 2,  "model": "2634-Umi_A2_O2000_J10_E1000_Ep1500_4.h5"},
        "Brightness":  {"ar": 2,  "model": "2620-Lum_A2_O2000_J10_E1000_Ep500_10.h5"},
        "Voltage":     {"ar": 2,  "model": "2602-Vol_A2_O2000_J10_E1000_Ep100_2.h5"},
    },
    "Conservative": {
        "description": "Intent-aware policy (MSE < 3x minimum)",
        "Temperature": {"ar": 4,  "model": "5416-Temp_A4_O2000_J10_E1000_Ep1000_1.h5"},
        "Humidity":    {"ar": 10, "model": "3620-Umi_A10_O2000_J10_E1000_Ep500_10.h5"},
        "Brightness":  {"ar": 6,  "model": "3420-Lum_A6_O2000_J10_E1000_Ep500_10.h5"},
        "Voltage":     {"ar": 60, "model": "4624-Vol_A60_O2000_J10_E1000_Ep1000_4.h5"},
    },
    "Aggressive": {
        "description": "Intent-aware policy (knee-point, max compression)",
        "Temperature": {"ar": 15, "model": "16834-Temp_A15_O2000_J10_E1000_Ep1500_7.h5"},
        "Humidity":    {"ar": 12, "model": "3825-Umi_A12_O2000_J10_E1000_Ep1000_5.h5"},
        "Brightness":  {"ar": 10, "model": "3630-Lum_A10_O2000_J10_E1000_Ep1000_10.h5"},
        "Voltage":     {"ar": 20, "model": "4221-Vol_A20_O2000_J10_E1000_Ep1000_1.h5"},
    },
}

# ============================================================
#  GLOBAL STATE
# ============================================================

class PipelineState:
    def __init__(self):
        self.lstm_model = None
        self.perceptron_models = {}
        self.device_registry = {}
        self.port_to_device = {}
        self.classified_devices = []
        self.fipa_messages = []
        self.mqtt_messages_edge = 0
        self.mqtt_messages_cloud = 0
        self.mqtt_bytes_edge = 0
        self.mqtt_bytes_cloud = 0
        self.current_scenario = None
        self.current_policy = None
        self.results = {}

        self.edge_client = None
        self.cloud_client = None

        # Phase 1: readings from unclassified sensors, per port
        self.discovery_buffer = {}

        # Phase 2: raw readings per topic (device_id)
        self.raw_data_buffer = {}

        # Reconstruction packages received from the cloud broker
        self.reconstruction_buffer = {}

        # Configuration received on iot/edge/config/hosts/#
        self.hosts_config_lines = []
        self.hosts_config_last_arrival = None

        # Phase 2 synchronization: one event per (label, scenario), set by the Reconstructor
        self.reconstruction_done = {}

        self.discovery_timestamps = {}
        self.operation_timestamps = {}
        self.fipa_bytes_total = 0

        # ---- concurrency and scalability instrumentation ----
        self.concurrency = 1            # maximum number of simultaneous devices (semaphore); 1 = serial, as in v11
        self.fixed_types = False        # if True, use the FoT-Simulator true_type instead of the LSTM prediction (scalability sweep)
        self.semaphore = None           # asyncio.Semaphore(concurrency), created in main
        self.recon_executor = None      # ThreadPoolExecutor for reconstruction (CPU-bound)
        self.model_locks = {}           # threading.Lock per model
        self._model_locks_guard = threading.Lock()

        self.inflight = 0               # devices currently being processed
        self.peak_inflight = 0          # peak concurrency actually reached
        self.samples = []               # time series: t, msgs, inflight, cpu, rss
        self.timeouts = []              # (label, scenario) pairs that timed out

    def model_lock(self, model_name):
        # one lock per model: concurrent predict calls on the same Keras object are not safe
        with self._model_locks_guard:
            if model_name not in self.model_locks:
                self.model_locks[model_name] = threading.Lock()
            return self.model_locks[model_name]

    def enter_inflight(self):
        self.inflight += 1
        if self.inflight > self.peak_inflight:
            self.peak_inflight = self.inflight

    def exit_inflight(self):
        self.inflight = max(0, self.inflight - 1)

    def log_fipa(self, sender, receiver, performative, size):
        self.fipa_messages.append({
            "scenario": self.current_scenario,
            "sender": sender, "receiver": receiver,
            "performative": performative, "payload_bytes": size,
            "timestamp": time.time(),
        })
        self.fipa_bytes_total += size

STATE = PipelineState()

# ============================================================
#  CONFIG: parse iot/edge/config/hosts
# ============================================================

def parse_hosts_config(lines):
    """    Convert the JSON payloads received on iot/edge/config/hosts/# into SIMULATED_PORTS.

        Each payload describes one sensor, e.g.
            {"type":"sensor","name":"h1","name_iot":"sc01",
             "ip":"10.0.0.1","sensorType":"temperature"}

        Returns a list of dicts with port, true_type and label, sorted by port.

    """
    ports = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "sensor":
            continue

        name_iot        = entry["name_iot"]           # "sc01"
        sensor_type_raw = entry["sensorType"]         # "temperature"
        sensor_type     = SENSOR_TYPE_MAP.get(
            sensor_type_raw.lower(), sensor_type_raw.capitalize())

        # "sc01" -> "port_01"
        port_num = name_iot.replace("sc", "")
        port     = f"port_{port_num}"
        label    = f"{name_iot.upper()} ({entry.get('name', '')})"

        ports.append({
            "port":      port,
            "true_type": sensor_type,
            "label":     label,
        })

    # Sort by port
    ports.sort(key=lambda p: p["port"])
    return ports

# ============================================================
#  MODEL
# ============================================================

def load_perceptron(model_path):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[PERCEPTRON_J])
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(
        learning_rate=1e-6, momentum=0.9))
    model.load_weights(model_path)
    return model


def aggregate_data(raw_data, ar):
    means = []
    for i in range(0, len(raw_data), ar):
        block = raw_data[i:i+ar]
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


def compute_mse(original, reconstructed):
    return float(np.mean((np.array(original) - np.array(reconstructed)) ** 2))


def compute_mae(original, reconstructed):
    return float(np.mean(np.abs(np.array(original) - np.array(reconstructed))))


# ============================================================
#  MQTT
# ============================================================

def _parse_fot_sim_value(raw_value):
    """Convert the 'value' field sent by the publisher into a list of floats."""
    if isinstance(raw_value, (int, float)):
        return [float(raw_value)]
    if isinstance(raw_value, list):
        return [float(v) for v in raw_value]
    raw_value = raw_value.strip()
    if raw_value.startswith('['):
        return [float(v) for v in ast.literal_eval(raw_value)]
    return [float(raw_value)]


def setup_mqtt():
    edge_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                              client_id="pipeline-edge")

    def on_edge_message(client, userdata, msg):
        topic = msg.topic

        # Host configuration: one message per sensor
        if topic.startswith("iot/edge/config/hosts"):
            STATE.hosts_config_lines.append(msg.payload.decode())
            STATE.hosts_config_last_arrival = time.time()
            return

        # Readings from unclassified sensors (Phase 1)
        if topic.startswith("iot/edge/unknown/"):
            payload = json.loads(msg.payload.decode())
            port = topic.split("/")[-1]
            if port not in STATE.discovery_buffer:
                STATE.discovery_buffer[port] = []
            values = _parse_fot_sim_value(payload["value"])
            STATE.discovery_buffer[port].extend(values)
            STATE.mqtt_messages_edge += len(values)
            return

        # Raw readings (Phase 2)
        if topic.startswith("iot/edge/raw/"):
            payload = json.loads(msg.payload.decode())
            if topic not in STATE.raw_data_buffer:
                STATE.raw_data_buffer[topic] = []
            values = _parse_fot_sim_value(payload["value"])
            STATE.raw_data_buffer[topic].extend(values)
            STATE.mqtt_messages_edge += len(values)
            STATE.mqtt_bytes_edge += len(msg.payload)
            return

    edge_client.on_message = on_edge_message
    edge_client.connect(EDGE_BROKER_HOST, EDGE_BROKER_PORT)
    edge_client.subscribe("iot/edge/config/hosts/#")
    edge_client.subscribe("iot/edge/unknown/#")
    # iot/edge/raw/# is subscribed per device after registration
    edge_client.loop_start()

    cloud_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                               client_id="pipeline-cloud")

    def on_cloud_message(client, userdata, msg):
        topic = msg.topic
        if topic.startswith("iot/cloud/compressed/"):
            payload = json.loads(msg.payload.decode())
            STATE.reconstruction_buffer[topic] = payload
            STATE.mqtt_messages_cloud += 1
            STATE.mqtt_bytes_cloud += len(msg.payload)

    cloud_client.on_message = on_cloud_message
    cloud_client.connect(CLOUD_BROKER_HOST, CLOUD_BROKER_PORT)
    cloud_client.subscribe("iot/cloud/compressed/#")
    cloud_client.loop_start()

    STATE.edge_client = edge_client
    STATE.cloud_client = cloud_client
    print("    Edge broker connected (localhost:1883)")
    print("    Cloud broker connected (localhost:1884)")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def publish_edge(topic, payload, retain=False):
    data = json.dumps(payload, cls=NumpyEncoder)
    STATE.edge_client.publish(topic, data, qos=0, retain=retain)
    STATE.mqtt_messages_edge += 1
    STATE.mqtt_bytes_edge += len(data)


def publish_cloud(topic, payload, retain=False):
    data = json.dumps(payload, cls=NumpyEncoder)
    STATE.cloud_client.publish(topic, data, qos=0, retain=retain)
    STATE.mqtt_messages_cloud += 1
    STATE.mqtt_bytes_cloud += len(data)


# ============================================================
#  AGENT 1: CLASSIFIER
# ============================================================

class ClassifierAgent(Agent):
    async def setup(self):
        pass

    def classify_window(self, window):
        x = np.array(window).reshape(1, WINDOW_SIZE, 1)
        prediction = STATE.lstm_model.predict(x, verbose=0)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_idx])
        return SENSOR_LABELS[class_idx], confidence


# ============================================================
#  AGENT 2: ORCHESTRATOR
# ============================================================

class OrchestratorAgent(Agent):
    async def setup(self):
        self.add_behaviour(OrchestrateBehaviour())

class OrchestrateBehaviour(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=5)
        if not msg:
            return
        try:
            data = json.loads(msg.body)
        except json.JSONDecodeError:
            return

        sensor_type = data.get("sensor_type")
        action = data.get("action", "discover")

        if action == "discover":
            reg_body = json.dumps({
                "action": "register_device",
                "sensor_type": sensor_type,
                "confidence": data.get("confidence", 0),
                "port": data.get("port", ""),
                "device_label": data.get("device_label", ""),
            })
            reg_msg = Message(to="registration@localhost")
            reg_msg.set_metadata("performative", "request")
            reg_msg.body = reg_body
            await self.send(reg_msg)
            STATE.log_fipa("orchestrator", "registration", "REQUEST", len(reg_body))

            pol_body = json.dumps({
                "action": "resolve_policy_only",
                "sensor_type": sensor_type,
                "port": data.get("port", ""),
                "device_label": data.get("device_label", ""),
            })
            pol_msg = Message(to="policy@localhost")
            pol_msg.set_metadata("performative", "request")
            pol_msg.body = pol_body
            await self.send(pol_msg)
            STATE.log_fipa("orchestrator", "policy", "REQUEST", len(pol_body))

        elif action == "operate":
            pol_body = json.dumps({
                "action": "resolve_and_apply_policy",
                "sensor_type": sensor_type,
                "port": data.get("port", ""),
                "num_observations": NUM_OBSERVATIONS,
                "device_label": data.get("device_label", ""),
            })
            pol_msg = Message(to="policy@localhost")
            pol_msg.set_metadata("performative", "request")
            pol_msg.body = pol_body
            await self.send(pol_msg)
            STATE.log_fipa("orchestrator", "policy", "REQUEST", len(pol_body))


# ============================================================
#  AGENT 3: REGISTRATION
# ============================================================

class RegistrationAgent(Agent):
    async def setup(self):
        self.add_behaviour(RegistrationBehaviour())

class RegistrationBehaviour(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=5)
        if not msg:
            return
        try:
            data = json.loads(msg.body)
        except json.JSONDecodeError:
            return
        if data.get("action") != "register_device":
            return

        device_id = str(uuid.uuid4())[:8]
        sensor_type = data["sensor_type"]
        port = data.get("port", "")

        raw_topic             = f"iot/edge/raw/{sensor_type.lower()}/{device_id}"
        compressed_topic      = f"iot/edge/compressed/{sensor_type.lower()}/{device_id}"
        cloud_compressed_topic= f"iot/cloud/compressed/{sensor_type.lower()}/{device_id}"
        reconstructed_topic   = f"iot/cloud/reconstructed/{sensor_type.lower()}/{device_id}"
        metrics_topic         = f"iot/cloud/metrics/{device_id}"

        device_info = {
            "device_id": device_id,
            "sensor_type": sensor_type.lower(),
            "port": port,
            "device_label": data.get("device_label", ""),
            "confidence": data.get("confidence", 0),
            "topics": {
                "raw": raw_topic,
                "compressed": compressed_topic,
                "cloud_compressed": cloud_compressed_topic,
                "reconstructed": reconstructed_topic,
                "metrics": metrics_topic,
            },
        }

        STATE.device_registry[device_id] = device_info
        STATE.port_to_device[port] = device_info

        # Retained: the publisher reads it and starts publishing on raw_topic
        publish_edge(f"iot/edge/registry/{port}", device_info, retain=True)

        # Subscribe to this device's raw topic to accumulate Phase 2 data
        STATE.edge_client.subscribe(raw_topic)

        inform_body = json.dumps({
            "action": "device_registered",
            "device_id": device_id,
            "sensor_type": sensor_type,
            "topics": device_info["topics"],
        })
        inform_msg = Message(to="orchestrator@localhost")
        inform_msg.set_metadata("performative", "inform")
        inform_msg.body = inform_body
        await self.send(inform_msg)
        STATE.log_fipa("registration", "orchestrator", "INFORM", len(inform_body))

        print(f"      [Registration] {device_id}: {sensor_type}")
        print(f"        MQTT raw:    {raw_topic}")
        print(f"        MQTT compr:  {compressed_topic}")
        print(f"        MQTT recon:  {reconstructed_topic}")


# ============================================================
#  AGENT 4: POLICY
# ============================================================

class PolicyAgent(Agent):
    async def setup(self):
        self.add_behaviour(PolicyBehaviour())

class PolicyBehaviour(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=5)
        if not msg:
            return
        try:
            data = json.loads(msg.body)
        except json.JSONDecodeError:
            return

        action = data.get("action")
        sensor_type = data["sensor_type"]
        policy = STATE.current_policy[sensor_type]
        ar = policy["ar"]
        model_name = policy["model"]

        if action == "resolve_policy_only":
            print(f"      [Policy] {sensor_type}: ar={ar}, model={model_name}")
            inform_body = json.dumps({
                "action": "policy_resolved",
                "sensor_type": sensor_type,
                "ar": ar,
                "model": model_name,
            })
            inform_msg = Message(to="orchestrator@localhost")
            inform_msg.set_metadata("performative", "inform")
            inform_msg.body = inform_body
            await self.send(inform_msg)
            STATE.log_fipa("policy", "orchestrator", "INFORM", len(inform_body))

        elif action == "resolve_and_apply_policy":
            port = data.get("port", "")
            device_info = STATE.port_to_device.get(port, {})

            print(f"      [Policy] {sensor_type}: ar={ar}, model={model_name}")

            agg_body = json.dumps({
                "action": "aggregate",
                "sensor_type": sensor_type,
                "ar": ar,
                "perceptron_model": model_name,
                "num_observations": data.get("num_observations", NUM_OBSERVATIONS),
                "device_label": data.get("device_label", ""),
                "device_info": device_info,
            })
            agg_msg = Message(to="aggregator@localhost")
            agg_msg.set_metadata("performative", "request")
            agg_msg.body = agg_body
            await self.send(agg_msg)
            STATE.log_fipa("policy", "aggregator", "REQUEST", len(agg_body))


# ============================================================
#  AGENT 5: AGGREGATOR
#  Consumes raw data from iot/edge/raw/{type}/{device_id}
# ============================================================

class DataAggregatorAgent(Agent):
    async def setup(self):
        self.add_behaviour(AggregatorBehaviour())

class AggregatorBehaviour(CyclicBehaviour):
    """Receive requests and start one worker per device, so that N aggregations
        can wait for data in parallel (a CyclicBehaviour cannot block in the wait loop).
    """
    async def run(self):
        msg = await self.receive(timeout=5)
        if not msg:
            return
        try:
            data = json.loads(msg.body)
        except json.JSONDecodeError:
            return
        if data.get("action") != "aggregate":
            return
        self.agent.add_behaviour(AggregateWorker(data))


class AggregateWorker(OneShotBehaviour):
    def __init__(self, data):
        super().__init__()
        self.data = data

    async def run(self):
        data = self.data
        sensor_type  = data["sensor_type"]
        ar           = data["ar"]
        num_obs      = data["num_observations"]
        model_name   = data["perceptron_model"]
        device_label = data.get("device_label", "")
        device_info  = data.get("device_info", {})
        raw_topic    = device_info.get("topics", {}).get("raw", "")
        scenario     = STATE.current_scenario  # read once at the start; the per-scenario barrier keeps it stable

        # Wait until NUM_OBSERVATIONS readings arrive via MQTT
        print(f"      [Aggregator] Aguardando {num_obs} leituras em {raw_topic} ...")
        wait_start = time.time()
        while len(STATE.raw_data_buffer.get(raw_topic, [])) < num_obs:
            await asyncio.sleep(0.5)
            elapsed = time.time() - wait_start
            if elapsed > 600.0:  # timeout: 10 min
                print(f"      [Aggregator] TIMEOUT aguardando dados de {raw_topic}")
                return

        raw = STATE.raw_data_buffer[raw_topic][:num_obs]
        n = len(raw)

        aggregated_means = aggregate_data(raw, ar)
        num_agg      = len(aggregated_means)
        raw_bytes    = n * 8
        agg_bytes    = num_agg * 8
        reduction_pct = round((1 - agg_bytes / raw_bytes) * 100, 2)
        seed = raw[:PERCEPTRON_J]

        print(f"      [Aggregator] {sensor_type}: {n} pts -> {num_agg} pts "
              f"(ar={ar}, {reduction_pct}% saved)")

        if device_info and "topics" in device_info:
            publish_edge(device_info["topics"]["compressed"], {
                "sensor_type": sensor_type,
                "ar": ar,
                "num_aggregated": num_agg,
                "reduction_pct": reduction_pct,
            })

            cloud_topic = device_info["topics"]["cloud_compressed"]

            reconstruction_package = {
                "sensor_type": sensor_type,
                "ar": ar,
                "perceptron_model": model_name,
                "num_original": n,
                "aggregated_means": aggregated_means,
                "seed": seed,
                "device_label": device_label,
                "device_id": device_info.get("device_id", ""),
                "scenario": scenario,
                "raw_bytes": raw_bytes,
                "agg_bytes": agg_bytes,
                "reduction_pct": reduction_pct,
            }
            publish_cloud(cloud_topic, reconstruction_package)

            recon_request = json.dumps({
                "action": "reconstruct",
                "sensor_type": sensor_type,
                "device_id": device_info.get("device_id", ""),
                "device_label": device_label,
                "perceptron_model": model_name,
                "ar": ar,
                "cloud_topic": cloud_topic,
                "raw_topic": raw_topic,
                "num_observations": num_obs,
                "raw_bytes": raw_bytes,
                "agg_bytes": agg_bytes,
                "reduction_pct": reduction_pct,
                "scenario": scenario,
                "agg_done_t": time.time(),
            })
            recon_msg = Message(to="reconstructor@localhost")
            recon_msg.set_metadata("performative", "request")
            recon_msg.body = recon_request
            await self.send(recon_msg)
            STATE.log_fipa("aggregator", "reconstructor", "REQUEST", len(recon_request))


# ============================================================
#  AGENT 6: RECONSTRUCTOR
#  Uses raw_data_buffer (the same buffer as the Aggregator) for MAE/MSE
# ============================================================

class ReconstructorAgent(Agent):
    async def setup(self):
        self.add_behaviour(ReconstructorBehaviour())

class ReconstructorBehaviour(CyclicBehaviour):
    """Start one worker per device to reconstruct in parallel."""
    async def run(self):
        msg = await self.receive(timeout=5)
        if not msg:
            return
        try:
            metadata = json.loads(msg.body)
        except json.JSONDecodeError:
            return
        if metadata.get("action") != "reconstruct":
            return
        self.agent.add_behaviour(ReconstructWorker(metadata))


def _reconstruct_blocking(model_name, aggregated_means, raw, ar):
    """Runs in a thread (CPU-bound). One lock per model, since concurrent predict
        calls on the same Keras object are not thread-safe. Same computation as v11.
    """
    model = STATE.perceptron_models[model_name]
    with STATE.model_lock(model_name):
        reconstructed = reconstruct_data(aggregated_means, raw, model, PERCEPTRON_J, ar)
    return reconstructed


class ReconstructWorker(OneShotBehaviour):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

    async def run(self):
        metadata = self.metadata
        sensor_type  = metadata["sensor_type"]
        model_name   = metadata["perceptron_model"]
        ar           = metadata["ar"]
        device_label = metadata.get("device_label", "")
        device_id    = metadata.get("device_id", "")
        cloud_topic  = metadata.get("cloud_topic", "")
        raw_topic    = metadata.get("raw_topic", "")
        num_obs      = metadata.get("num_observations", NUM_OBSERVATIONS)
        scenario     = metadata.get("scenario", STATE.current_scenario)

        # Wait for the reconstruction package in the MQTT buffer
        max_wait = 5.0
        wait_start = time.time()
        while cloud_topic not in STATE.reconstruction_buffer:
            await asyncio.sleep(0.1)
            if time.time() - wait_start > max_wait:
                print(f"      [Reconstructor] TIMEOUT waiting for {cloud_topic}")
                return

        package          = STATE.reconstruction_buffer.pop(cloud_topic)
        aggregated_means = package["aggregated_means"]
        n                = package["num_original"]

        # Original data from the buffer used by the Aggregator
        raw = STATE.raw_data_buffer.get(raw_topic, [])[:num_obs]
        if len(raw) < n:
            print(f"      [Reconstructor] Dados insuficientes em {raw_topic}")
            return

        recon_start = time.time()
        # CPU-bound: offloaded to the thread pool so the event loop is not blocked
        # (under concurrency, this is where the CPU saturates)
        loop = asyncio.get_running_loop()
        reconstructed = await loop.run_in_executor(
            STATE.recon_executor, _reconstruct_blocking,
            model_name, aggregated_means, raw, ar)
        recon_time_ms = (time.time() - recon_start) * 1000

        mse = compute_mse(raw[:n], reconstructed)
        mae = compute_mae(raw[:n], reconstructed)

        print(f"      [Reconstructor] {sensor_type}: MAE={mae:.6f} MSE={mse:.6f} "
              f"({recon_time_ms:.0f}ms)")

        recon_topic = (f"iot/cloud/reconstructed/{sensor_type.lower()}/"
                       f"{device_id}/{scenario}")
        publish_cloud(recon_topic, {
            "sensor_type": sensor_type,
            "scenario": scenario,
            "num_points": len(reconstructed),
            "sample_original": raw[:n],
            "sample_aggregated": expand_aggregated(aggregated_means, ar, n),
            "sample_reconstructed": reconstructed,
        }, retain=True)

        metrics_topic = f"iot/cloud/metrics/{device_id}/{scenario}"
        publish_cloud(metrics_topic, {
            "sensor_type": sensor_type,
            "device_id": device_id,
            "scenario": scenario,
            "ar": ar,
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "reduction_pct": metadata.get("reduction_pct", 0),
        }, retain=True)

        if scenario not in STATE.results:
            STATE.results[scenario] = {}
        STATE.results[scenario][device_label] = {
            "sensor_type": sensor_type,
            "ar": ar,
            "model": model_name,
            "raw_bytes": metadata["raw_bytes"],
            "agg_bytes": metadata["agg_bytes"],
            "reduction_pct": metadata["reduction_pct"],
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "reconstruction_time_ms": round(recon_time_ms, 2),
        }

        # timing for makespan and end-to-end latency
        key = f"{device_label}__{scenario}"
        ts = STATE.operation_timestamps.setdefault(key, {})
        ts.update({
            "label": device_label, "scenario": scenario, "sensor_type": sensor_type,
            "agg_done_t": metadata.get("agg_done_t"),
            "recon_start_t": recon_start, "recon_done_t": time.time(),
            "recon_ms": round(recon_time_ms, 2),
        })

        # Signal the main loop that this device/scenario is complete
        event_key = f"{device_label}__{scenario}"
        if event_key in STATE.reconstruction_done:
            STATE.reconstruction_done[event_key].set()


# ============================================================
#  MAIN
# ============================================================

async def main():
    print("=" * 70)
    print("  MQTT-INTEGRATED AGENTIC IoT PIPELINE (v12)")
    print(f"  Concurrency C={STATE.concurrency} | "
          f"classification={'FIXED (true types)' if STATE.fixed_types else 'LSTM'}")
    print("  Sensores: dinâmico via iot/edge/config/hosts")
    print("  Dados Fase 2: iot/edge/raw/{type}/{device_id} (tempo real)")
    print("=" * 70)
    print()

    print("  Loading LSTM classifier...")
    STATE.lstm_model = tf.keras.models.load_model("model_classifier.h5")
    print("  Warming up LSTM model...")
    STATE.lstm_model.predict(np.zeros((1, WINDOW_SIZE, 1)), verbose=0)
    print("  LSTM model ready.")

    all_models = set()
    for scenario in POLICY_SCENARIOS.values():
        for sensor in SENSOR_TYPES:
            all_models.add(scenario[sensor]["model"])
    print(f"  Loading {len(all_models)} perceptron models...")
    for model_file in sorted(all_models):
        if os.path.exists(model_file):
            STATE.perceptron_models[model_file] = load_perceptron(model_file)
        else:
            print(f"    MISSING: {model_file}")
            return

    print("\n  Connecting MQTT brokers...")
    setup_mqtt()

    # Wait for the host configuration (iot/edge/config/hosts/#, retained per sensor)
    print("\n  Aguardando iot/edge/config/hosts/# ...")
    wait_start = time.time()
    while True:
        await asyncio.sleep(0.2)
        # Complete when at least one message has arrived
        # and no new message arrives for 2 seconds
        if (STATE.hosts_config_lines and
                STATE.hosts_config_last_arrival is not None and
                time.time() - STATE.hosts_config_last_arrival > 2.0):
            break
        if time.time() - wait_start > 30.0:
            print("  ERRO: timeout aguardando iot/edge/config/hosts/#. "
                  "O FoT-Sim está publicando com retain=True?")
            return

    SIMULATED_PORTS = parse_hosts_config(STATE.hosts_config_lines)
    total = len(SIMULATED_PORTS)
    print(f"  Configuração recebida: {total} sensor(es)")
    for p in SIMULATED_PORTS:
        print(f"    {p['port']} -> {p['true_type']} ({p['label']})")

    print("\n  Starting 6 SPADE agents...")
    classifier  = ClassifierAgent("classifier@localhost",   "classifier")
    orchestrator= OrchestratorAgent("orchestrator@localhost","orchestrator")
    registration= RegistrationAgent("registration@localhost","registration")
    policy_agent= PolicyAgent("policy@localhost",           "policy")
    aggregator  = DataAggregatorAgent("aggregator@localhost","aggregator")
    reconstructor=ReconstructorAgent("reconstructor@localhost","reconstructor")

    await classifier.start()
    await orchestrator.start()
    await registration.start()
    await policy_agent.start()
    await aggregator.start()
    await reconstructor.start()
    print("  All 6 agents running.\n")

    STATE.current_scenario = "Discovery"
    STATE.current_policy   = POLICY_SCENARIOS["Conservative"]

    # ================================================================
    #  PHASE 1: DISCOVERY
    # ================================================================
    print("=" * 70)
    print(f"  PHASE 1: DISCOVERY ({total} sensor(es))")
    print("=" * 70)
    print()

    discovery_start_time = time.time()

    for i, port_info in enumerate(SIMULATED_PORTS, 1):
        port      = port_info["port"]
        true_type = port_info["true_type"]
        label     = port_info["label"]

        print(f"    Device {i}/{total}: {label}")
        device_discovery_start = time.time()

        print(f"      [Discovery] Aguardando dados em iot/edge/unknown/{port} ...")
        wait_start = time.time()
        while len(STATE.discovery_buffer.get(port, [])) < WINDOW_SIZE:
            await asyncio.sleep(0.2)
            if time.time() - wait_start > 30.0:
                print(f"      [Discovery] TIMEOUT: sem dados de {port}. Pulando.")
                break

        readings = STATE.discovery_buffer.get(port, [])
        if len(readings) < WINDOW_SIZE:
            print()
            continue

        window = readings[:WINDOW_SIZE]
        classify_start = time.time()
        predicted, confidence = classifier.classify_window(window)
        classify_time = (time.time() - classify_start) * 1000
        lstm_predicted = predicted  # raw LSTM prediction

        # --fixed-types: the LSTM still runs (its latency is measured), but the policy
        # uses the true type from the FoT-Simulator, so that every run processes
        # the same workload, without classification variance.
        if STATE.fixed_types:
            predicted = true_type

        status = "OK" if predicted == true_type else "XX"
        tag = (f"[FIXED:{true_type}] (LSTM disse {lstm_predicted})"
               if STATE.fixed_types and lstm_predicted != true_type
               else f"[{status}]")
        print(f"      [Classifier] {predicted} ({confidence:.4f}) {tag} "
              f"({classify_time:.1f}ms)")

        STATE.classified_devices.append({
            "port": port,
            "sensor_type": predicted,
            "lstm_predicted": lstm_predicted,
            "true_type": true_type,
            "lstm_correct": (lstm_predicted == true_type),
            "confidence": confidence,
            "label": label,
        })

        body = json.dumps({
            "action": "discover",
            "sensor_type": predicted,
            "confidence": confidence,
            "port": port,
            "device_label": label,
        })

        class SendDiscovery(OneShotBehaviour):
            def __init__(self, body_text):
                super().__init__()
                self.body_text = body_text
            async def run(self):
                msg = Message(to="orchestrator@localhost")
                msg.set_metadata("performative", "inform")
                msg.body = self.body_text
                await self.send(msg)

        classifier.add_behaviour(SendDiscovery(body))
        STATE.log_fipa("classifier", "orchestrator", "INFORM", len(body))

        # Wait for the registration to complete (port_to_device filled by the
        # RegistrationAgent) instead of a fixed sleep.
        reg_wait = time.time()
        while port not in STATE.port_to_device:
            await asyncio.sleep(0.05)
            if time.time() - reg_wait > 30.0:
                print(f"      [Discovery] TIMEOUT no registro de {port}.")
                break

        STATE.discovery_timestamps[port] = {
            "label": label,
            "sensor_type": predicted,
            "classify_ms": round(classify_time, 2),
            "total_ms": round((time.time() - device_discovery_start) * 1000, 2),
        }
        print()

    discovery_total_time = time.time() - discovery_start_time
    print(f"  Phase 1 complete: {len(STATE.device_registry)} devices registered.")
    print(f"  Discovery time: {discovery_total_time:.1f}s")
    print(f"  FIPA-ACL messages (discovery): {len(STATE.fipa_messages)}")
    discovery_msg_count = len(STATE.fipa_messages)

    # ================================================================
    #  PHASE 2: OPERATION  (concurrency limited by a semaphore)
    # ================================================================
    STATE.semaphore = asyncio.Semaphore(STATE.concurrency)

    # --- load sampler: runs during the whole Phase 2 ---
    sampler_stop = asyncio.Event()

    async def load_sampler(interval=0.5):
        t0 = time.time()
        last_edge, last_t = STATE.mqtt_messages_edge, t0
        while not sampler_stop.is_set():
            await asyncio.sleep(interval)
            now = time.time()
            d_edge = STATE.mqtt_messages_edge - last_edge
            rate = d_edge / max(now - last_t, 1e-6)
            cpu = _PROC.cpu_percent(interval=None) if _PROC else None
            rss = (_PROC.memory_info().rss / (1024*1024)) if _PROC else None
            STATE.samples.append({
                "t": round(now - t0, 3),
                "edge_msgs": STATE.mqtt_messages_edge,
                "cloud_msgs": STATE.mqtt_messages_cloud,
                "edge_rate_msgs_s": round(rate, 1),
                "inflight": STATE.inflight,
                "cpu_pct": cpu, "rss_mb": round(rss, 1) if rss else None,
            })
            last_edge, last_t = STATE.mqtt_messages_edge, now

    sampler_task = asyncio.create_task(load_sampler())

    async def run_device(i, dev, scenario_name, total):
        """Complete flow of one device under the semaphore (concurrency <= C)."""
        port        = dev["port"]
        sensor_type = dev["sensor_type"]
        label       = dev["label"]
        async with STATE.semaphore:
            STATE.enter_inflight()
            device_op_start = time.time()
            print(f"    [dispatch {i}/{total}] {label} ({scenario_name})")

            body = json.dumps({
                "action": "operate",
                "sensor_type": sensor_type,
                "port": port,
                "device_label": label,
            })

            class SendOperate(OneShotBehaviour):
                def __init__(self, body_text):
                    super().__init__()
                    self.body_text = body_text
                async def run(self):
                    msg = Message(to="orchestrator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.body = self.body_text
                    await self.send(msg)

            event_key = f"{label}__{scenario_name}"
            event = asyncio.Event()
            STATE.reconstruction_done[event_key] = event

            classifier.add_behaviour(SendOperate(body))
            STATE.log_fipa("classifier", "orchestrator", "INFORM", len(body))

            timed_out = False
            try:
                await asyncio.wait_for(event.wait(), timeout=1200.0)
            except asyncio.TimeoutError:
                timed_out = True
                STATE.timeouts.append(event_key)
                print(f"      [Pipeline] TIMEOUT aguardando reconstrução de {label}")

            ts = STATE.operation_timestamps.setdefault(event_key, {})
            ts.update({
                "label": label, "scenario": scenario_name, "sensor_type": sensor_type,
                "dispatch_t": device_op_start, "complete_t": time.time(),
                "e2e_ms": round((time.time() - device_op_start) * 1000, 2),
                "timed_out": timed_out,
            })
            STATE.exit_inflight()

    for scenario_name, scenario_config in POLICY_SCENARIOS.items():
        STATE.current_scenario = scenario_name
        STATE.current_policy   = scenario_config
        scenario_start_time    = time.time()

        print(f"\n  {'=' * 60}")
        print(f"  PHASE 2 - SCENARIO: {scenario_name}  (C={STATE.concurrency})")
        print(f"  {scenario_config['description']}")
        print(f"  {'=' * 60}")

        # Dispatch all devices; the semaphore allows at most C at a time.
        tasks = [
            asyncio.create_task(run_device(i, dev, scenario_name, total))
            for i, dev in enumerate(STATE.classified_devices, 1)
        ]
        await asyncio.gather(*tasks)

        # Makespan = max(complete) - min(dispatch)
        keys = [f"{d['label']}__{scenario_name}" for d in STATE.classified_devices]
        ds = [STATE.operation_timestamps[k]["dispatch_t"] for k in keys
              if k in STATE.operation_timestamps]
        cs = [STATE.operation_timestamps[k]["complete_t"] for k in keys
              if k in STATE.operation_timestamps]
        makespan = (max(cs) - min(ds)) if ds and cs else 0.0
        scenario_time = time.time() - scenario_start_time
        STATE.operation_timestamps.setdefault("_makespan", {})[scenario_name] = round(makespan, 2)
        await asyncio.sleep(2)
        print(f"\n    Scenario {scenario_name} complete. "
              f"(wall={scenario_time:.1f}s | makespan={makespan:.1f}s | "
              f"peak_inflight={STATE.peak_inflight})")

    sampler_stop.set()
    await sampler_task

    # ================================================================
    #  RESULTS
    # ================================================================
    print()
    print("=" * 70)
    print("  PIPELINE RESULTS")
    print("=" * 70)

    print(f"\n  DEVICE REGISTRY ({len(STATE.device_registry)} devices)")
    print(f"  {'ID':<10} {'Type':<12} {'Port':<10} {'Label':<25} {'Raw Topic'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*25} {'-'*40}")
    for dev_id, info in STATE.device_registry.items():
        print(f"  {dev_id:<10} {info['sensor_type']:<12} {info['port']:<10} "
              f"{info['device_label']:<25} {info['topics']['raw']}")

    for scenario_name in POLICY_SCENARIOS:
        scenario_results = STATE.results.get(scenario_name, {})
        print(f"\n  SCENARIO: {scenario_name}")
        print(f"  {POLICY_SCENARIOS[scenario_name]['description']}")
        print(f"  {'Device':<25} {'Type':<12} {'ar':<5} {'Saved%':>7} "
              f"{'MAE':>12} {'MSE':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*5} {'-'*7} {'-'*12} {'-'*12}")

        total_raw, total_agg, maes, mses = 0, 0, [], []
        for dev in STATE.classified_devices:
            lbl = dev["label"]
            r   = scenario_results.get(lbl)
            if r:
                total_raw += r["raw_bytes"]
                total_agg += r["agg_bytes"]
                maes.append(r["mae"])
                mses.append(r["mse"])
                print(f"  {lbl:<25} {r['sensor_type']:<12} {r['ar']:<5} "
                      f"{r['reduction_pct']:>6.1f}% {r['mae']:>12.6f} "
                      f"{r['mse']:>12.6f}")
        if maes:
            total_pct = (1 - total_agg / total_raw) * 100 if total_raw > 0 else 0
            print(f"  {'TOTAL/AVG':<25} {'':>12} {'':>5} "
                  f"{total_pct:>6.1f}% {np.mean(maes):>12.6f} "
                  f"{np.mean(mses):>12.6f}")

    print(f"\n  {'=' * 60}")
    print(f"  SCENARIO COMPARISON SUMMARY")
    print(f"  {'=' * 60}")
    print(f"\n  {'Scenario':<15} {'Traffic Saved':>15} {'Avg MAE':>12} {'Avg MSE':>12}")
    print(f"  {'-'*15} {'-'*15} {'-'*12} {'-'*12}")
    for scenario_name in POLICY_SCENARIOS:
        sr = STATE.results.get(scenario_name, {})
        if sr:
            tr  = sum(r["raw_bytes"] for r in sr.values())
            ta  = sum(r["agg_bytes"] for r in sr.values())
            pct = (1 - ta / tr) * 100 if tr > 0 else 0
            print(f"  {scenario_name:<15} {pct:>14.1f}% "
                  f"{np.mean([r['mae'] for r in sr.values()]):>12.6f} "
                  f"{np.mean([r['mse'] for r in sr.values()]):>12.6f}")

    print(f"\n  MESSAGING SUMMARY")
    print(f"  {'Protocol':<20} {'Messages':>10} {'Bytes':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    print(f"  {'FIPA-ACL':<20} {len(STATE.fipa_messages):>10} "
          f"{STATE.fipa_bytes_total:>12}")
    print(f"  {'MQTT (Edge)':<20} {STATE.mqtt_messages_edge:>10} "
          f"{STATE.mqtt_bytes_edge:>12}")
    print(f"  {'MQTT (Cloud)':<20} {STATE.mqtt_messages_cloud:>10} "
          f"{STATE.mqtt_bytes_cloud:>12}")
    total_msgs  = (len(STATE.fipa_messages) + STATE.mqtt_messages_edge
                   + STATE.mqtt_messages_cloud)
    total_bytes = (STATE.fipa_bytes_total + STATE.mqtt_bytes_edge
                   + STATE.mqtt_bytes_cloud)
    print(f"  {'TOTAL':<20} {total_msgs:>10} {total_bytes:>12}")

    print(f"\n  FIPA-ACL MESSAGE FLOW")
    flow_counts, flow_bytes = {}, {}
    for m in STATE.fipa_messages:
        route = f"{m['sender']} -> {m['receiver']} ({m['performative']})"
        flow_counts[route] = flow_counts.get(route, 0) + 1
        flow_bytes[route]  = flow_bytes.get(route, 0) + m['payload_bytes']
    for route, count in sorted(flow_counts.items()):
        print(f"    {route:<45} {count:>5} msgs  {flow_bytes[route]:>8} bytes")

    print(f"\n  DISCOVERY LATENCY (per device)")
    print(f"  {'Device':<25} {'Type':<12} {'Classify':>10} {'Total':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10}")
    classify_times, discovery_totals = [], []
    for port, ts in STATE.discovery_timestamps.items():
        classify_times.append(ts['classify_ms'])
        discovery_totals.append(ts['total_ms'])
        print(f"  {ts['label']:<25} {ts['sensor_type']:<12} "
              f"{ts['classify_ms']:>8.1f}ms {ts['total_ms']:>8.1f}ms")
    if classify_times:
        print(f"  {'AVERAGE':<25} {'':>12} "
              f"{np.mean(classify_times):>8.1f}ms "
              f"{np.mean(discovery_totals):>8.1f}ms")

    print(f"\n  CONTROL OVERHEAD")
    control_bytes = STATE.fipa_bytes_total
    data_bytes    = STATE.mqtt_bytes_edge + STATE.mqtt_bytes_cloud
    overhead_pct  = (control_bytes / (control_bytes + data_bytes) * 100
                     if (control_bytes + data_bytes) > 0 else 0)
    print(f"  FIPA-ACL control traffic:    {control_bytes:>10} bytes "
          f"({control_bytes/1024:.1f} KB)")
    print(f"  MQTT data traffic:           {data_bytes:>10} bytes "
          f"({data_bytes/1024:.1f} KB)")
    print(f"  Control overhead:            {overhead_pct:>9.2f}%")
    print(f"  Control msgs per device:     "
          f"{len(STATE.fipa_messages)/max(len(STATE.device_registry),1):>9.1f}")

    print(f"\n  TRAFFIC REDUCTION")
    print(f"  {'Scenario':<15} {'Raw bytes':>12} {'Agg bytes':>12} "
          f"{'Saved':>8} {'Ratio':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")
    for scenario_name in POLICY_SCENARIOS:
        sr = STATE.results.get(scenario_name, {})
        if sr:
            rt  = sum(r["raw_bytes"] for r in sr.values())
            at  = sum(r["agg_bytes"] for r in sr.values())
            sp  = (1 - at / rt) * 100 if rt > 0 else 0
            rat = rt / at if at > 0 else 0
            print(f"  {scenario_name:<15} {rt:>12} {at:>12} "
                  f"{sp:>6.1f}% {rat:>7.1f}x")

    # ================================================================
    #  SCALABILITY (concurrency, makespan, latency, load)
    # ================================================================
    def _pct(vals, p):
        if not vals:
            return 0.0
        return float(np.percentile(np.array(vals), p))

    e2e_all = [v["e2e_ms"] for k, v in STATE.operation_timestamps.items()
               if isinstance(v, dict) and "e2e_ms" in v]
    recon_all = [v["recon_ms"] for k, v in STATE.operation_timestamps.items()
                 if isinstance(v, dict) and "recon_ms" in v]
    makespans = STATE.operation_timestamps.get("_makespan", {})
    peak_rate = max((s["edge_rate_msgs_s"] for s in STATE.samples), default=0.0)
    peak_cpu  = max((s["cpu_pct"] for s in STATE.samples if s["cpu_pct"] is not None), default=None)
    peak_rss  = max((s["rss_mb"] for s in STATE.samples if s["rss_mb"] is not None), default=None)

    scalability = {
        "concurrency_limit_C": STATE.concurrency,
        "peak_inflight": STATE.peak_inflight,
        "makespan_per_scenario_s": makespans,
        "makespan_total_s": round(sum(makespans.values()), 2) if makespans else 0.0,
        "e2e_latency_ms": {
            "p50": round(_pct(e2e_all, 50), 1),
            "p95": round(_pct(e2e_all, 95), 1),
            "p99": round(_pct(e2e_all, 99), 1),
            "max": round(max(e2e_all), 1) if e2e_all else 0.0,
        },
        "reconstruction_ms": {
            "p50": round(_pct(recon_all, 50), 1),
            "p95": round(_pct(recon_all, 95), 1),
            "max": round(max(recon_all), 1) if recon_all else 0.0,
        },
        "broker_peak_ingress_msgs_s": round(peak_rate, 1),
        "cpu_peak_pct": round(peak_cpu, 1) if peak_cpu is not None else None,
        "rss_peak_mb": round(peak_rss, 1) if peak_rss is not None else None,
        "timeouts": STATE.timeouts,
        "num_timeouts": len(STATE.timeouts),
        "load_samples": STATE.samples,
    }

    print(f"\n  SCALABILITY (C={STATE.concurrency})")
    print(f"  Peak concurrency reached:   {STATE.peak_inflight}")
    print(f"  Makespan total:             {scalability['makespan_total_s']}s "
          f"(por cenário: {makespans})")
    print(f"  E2E latency  p50/p95/p99:   {scalability['e2e_latency_ms']['p50']} / "
          f"{scalability['e2e_latency_ms']['p95']} / "
          f"{scalability['e2e_latency_ms']['p99']} ms")
    print(f"  Broker peak ingress:        {scalability['broker_peak_ingress_msgs_s']} msgs/s")
    print(f"  CPU peak / RSS peak:        {scalability['cpu_peak_pct']}% / "
          f"{scalability['rss_peak_mb']} MB")
    print(f"  Timeouts (perda/backpressure): {len(STATE.timeouts)}")

    class NumpyEncoderFinal(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    output = {
        "pipeline_version": "v12_concurrent_scalability",
        "num_sensors": total,
        "concurrency_C": STATE.concurrency,
        "classification_mode": "fixed_types" if STATE.fixed_types else "lstm",
        "brokers": {
            "edge":  f"{EDGE_BROKER_HOST}:{EDGE_BROKER_PORT}",
            "cloud": f"{CLOUD_BROKER_HOST}:{CLOUD_BROKER_PORT}",
        },
        "device_registry": STATE.device_registry,
        "scenarios": STATE.results,
        "messaging": {
            "fipa_acl_total":     len(STATE.fipa_messages),
            "fipa_acl_bytes":     STATE.fipa_bytes_total,
            "fipa_acl_discovery": discovery_msg_count,
            "fipa_acl_operation": len(STATE.fipa_messages) - discovery_msg_count,
            "mqtt_edge_messages": STATE.mqtt_messages_edge,
            "mqtt_edge_bytes":    STATE.mqtt_bytes_edge,
            "mqtt_cloud_messages":STATE.mqtt_messages_cloud,
            "mqtt_cloud_bytes":   STATE.mqtt_bytes_cloud,
            "total_messages":     total_msgs,
            "total_bytes":        total_bytes,
            "control_overhead_pct": round(overhead_pct, 2),
        },
        "network_metrics": {
            "discovery_latency":      STATE.discovery_timestamps,
            "discovery_total_time_s": round(discovery_total_time, 2),
            "avg_classify_ms": round(np.mean(classify_times), 2)
                               if classify_times else 0,
            "operation_timestamps":   STATE.operation_timestamps,
        },
        "scalability": scalability,
    }
    mode_tag = "fixed" if STATE.fixed_types else "lstm"
    out_name = f"mqtt_pipeline_results_v12_N{total}_C{STATE.concurrency}_{mode_tag}.json"
    with open(out_name, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoderFinal)
    print(f"\n  Results saved to: {out_name}")

    STATE.edge_client.loop_stop()
    STATE.edge_client.disconnect()
    STATE.cloud_client.loop_stop()
    STATE.cloud_client.disconnect()

    await classifier.stop()
    await orchestrator.stop()
    await registration.stop()
    await policy_agent.stop()
    await aggregator.stop()
    await reconstructor.stop()

    if STATE.recon_executor is not None:
        STATE.recon_executor.shutdown(wait=True)

    print()
    print("=" * 70)
    print(f"  MQTT pipeline v12 complete (C={STATE.concurrency}).")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agentic IoT pipeline v12 (concurrent scalability)")
    parser.add_argument("--concurrency", "-c", type=int, default=1,
                        help="Máx de devices processados em paralelo na Fase 2 "
                             "(1 = serial/v11; use N para paralelismo total).")
    parser.add_argument("--sensor", type=int, default=None,
                        help="Inerte aqui (a contagem vem do FoT-Sim via MQTT); "
                             "aceito por compatibilidade de linha de comando.")
    parser.add_argument("--fixed-types", action="store_true",
                        help="Usa o tipo verdadeiro do FoT-Sim na política (o LSTM "
                             "ainda roda p/ medir latência, mas não decide). Isola o "
                             "eixo de escalabilidade: workload idêntico entre runs.")
    parser.add_argument("--recon-workers", type=int, default=None,
                        help="Tamanho do pool de threads de reconstrução "
                             "(default = concurrency).")
    args = parser.parse_args()

    STATE.concurrency = max(1, args.concurrency)
    STATE.fixed_types = args.fixed_types
    workers = args.recon_workers or STATE.concurrency
    STATE.recon_executor = ThreadPoolExecutor(max_workers=max(1, workers))

    asyncio.run(main())