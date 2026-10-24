"""Virtual sensor publisher for local runs of run_mqtt_pipeline_v11.py.

Replaces the FoT-Simulator in local experiments by publishing, on the edge
broker, the same messages that the simulator publishes:

1. One retained message per sensor on iot/edge/config/hosts/<name_iot>, from
   which the pipeline builds its sensor list:
       {"type":"sensor","name":"h1","name_iot":"sc01",
        "ip":"10.0.0.1","sensorType":"temperature"}
2. A discovery window of WINDOW_SIZE readings per sensor on
   iot/edge/unknown/<port>.
3. After the RegistrationAgent publishes the retained registry message on
   iot/edge/registry/<port>, the operation block of NUM_OBSERVATIONS readings
   on the raw topic created by the pipeline,
   iot/edge/raw/<sensor_type>/<device_id>.

Each sensor reads a fixed segment of one column of the Intel Berkeley Research
Lab data.txt (fields 4 to 7: temperature, humidity, light, voltage), starting
at the position given in ALL_PORTS. The discovery window is the first
WINDOW_SIZE readings of the same segment.

Run order:
  1) start the edge broker (Mosquitto, localhost:1883);
  2) start this publisher, which publishes the retained configuration first;
  3) start the pipeline.

  python sensorVirtual.py --dataset data.txt --sensors 8
"""

import argparse
import json
import threading
import time

import paho.mqtt.client as mqtt

# ============================================================
#  CONFIG
# ============================================================

EDGE_BROKER_HOST = "localhost"
EDGE_BROKER_PORT = 1883

NUM_OBSERVATIONS = 1000
WINDOW_SIZE = 3
MAX_POSITION = 22000

# The eight sensors of the functional experiment, with fixed positions in data.txt.
# 'name_iot' and 'name' follow the config/hosts schema read by the pipeline.
ALL_PORTS = [
    {"port": "port_01", "true_type": "Temperature", "position": 0,    "name_iot": "sc01", "name": "h1", "label": "Sensor-A"},
    {"port": "port_02", "true_type": "Temperature", "position": 500,  "name_iot": "sc02", "name": "h2", "label": "Sensor-B"},
    {"port": "port_03", "true_type": "Temperature", "position": 1000, "name_iot": "sc03", "name": "h3", "label": "Sensor-C"},
    {"port": "port_04", "true_type": "Humidity",    "position": 0,    "name_iot": "sc04", "name": "h4", "label": "Sensor-D"},
    {"port": "port_05", "true_type": "Humidity",    "position": 500,  "name_iot": "sc05", "name": "h5", "label": "Sensor-E"},
    {"port": "port_06", "true_type": "Brightness",  "position": 0,    "name_iot": "sc06", "name": "h6", "label": "Sensor-F"},
    {"port": "port_07", "true_type": "Brightness",  "position": 500,  "name_iot": "sc07", "name": "h7", "label": "Sensor-G"},
    {"port": "port_08", "true_type": "Voltage",     "position": 0,    "name_iot": "sc08", "name": "h8", "label": "Sensor-H"},
]

COLUMN_INDEX = {"Temperature": 4, "Humidity": 5, "Brightness": 6, "Voltage": 7}

# The pipeline maps sensorType in {temperature, humidity, light, voltage};
# Brightness is announced as "light".
TRUE_TYPE_TO_FOT = {
    "Temperature": "temperature",
    "Humidity":    "humidity",
    "Brightness":  "light",
    "Voltage":     "voltage",
}


# ============================================================
#  DATASET
# ============================================================

def load_columns(path, max_lines):
    cols = {"Temperature": [], "Humidity": [], "Brightness": [], "Voltage": []}
    count = 0
    with open(path, "r") as f:
        for line in f:
            if count >= max_lines:
                break
            parts = line.strip().split()
            if len(parts) >= 8:
                try:
                    cols["Temperature"].append(float(parts[4]))
                    cols["Humidity"].append(float(parts[5]))
                    cols["Brightness"].append(float(parts[6]))
                    cols["Voltage"].append(float(parts[7]))
                    count += 1
                except (ValueError, IndexError):
                    continue
    return cols


def segment_for(cols, true_type, position, n):
    column = cols[true_type]
    seg = column[position:position + n]
    if len(seg) < n:
        raise ValueError(
            f"Dataset too short for {true_type} at position {position}: "
            f"need {n} points, have {len(seg)}."
        )
    return seg


# ============================================================
#  SHARED STATE (publisher side)
# ============================================================

class PubState:
    def __init__(self):
        self.client = None
        self.segments = {}          # port -> list[float] (operation block)
        self.port_by_label = {}     # device_label -> port
        # device_label -> raw topic learned from the registry message
        self.raw_topic_by_port = {}
        self.lock = threading.Lock()

STATE = PubState()


# ============================================================
#  MQTT CALLBACKS
# ============================================================

def on_message(client, userdata, msg):
    """Store the raw topic announced in each registry message."""
    topic = msg.topic
    if topic.startswith("iot/edge/registry/"):
        try:
            info = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            return
        port = info.get("port", "")
        raw_topic = info.get("topics", {}).get("raw", "")
        if port and raw_topic:
            with STATE.lock:
                STATE.raw_topic_by_port[port] = raw_topic
            print(f"  [registry] {port} -> raw topic {raw_topic}")


def make_client():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="virtual-sensors-v11")
    client.on_message = on_message
    client.connect(EDGE_BROKER_HOST, EDGE_BROKER_PORT)
    client.subscribe("iot/edge/registry/#")
    client.loop_start()
    return client


# ============================================================
#  PUBLISH HELPERS
# ============================================================

def publish_config(client, ports):
    """Publish one retained config/hosts message per sensor."""
    for p in ports:
        payload = {
            "type": "sensor",
            "name": p["name"],
            "name_iot": p["name_iot"],
            "ip": "10.0.0." + p["port"].split("_")[1].lstrip("0"),
            "sensorType": TRUE_TYPE_TO_FOT[p["true_type"]],
        }
        # one retained message per sensor
        client.publish(f"iot/edge/config/hosts/{p['name_iot']}",
                       json.dumps(payload), qos=0, retain=True)
        print(f"  [config] {p['name_iot']} -> {payload['sensorType']} ({p['label']})")


def publish_discovery(client, port, window):
    payload = {
        "port": port,
        "value": str([float(v) for v in window]),
        "timestamp": str(time.perf_counter()),
    }
    client.publish(f"iot/edge/unknown/{port}", json.dumps(payload), qos=0)


def publish_operation_block(client, raw_topic, values):
    """Publish the whole operation block in a single message, once per sensor,
        as the FoT-Simulator does (it publishes the 1000 accumulated values at once).
        The pipeline reads the same buffer in the three policy scenarios.
    """
    payload = {
        "value": str([float(v) for v in values]),
        "timestamp": str(time.perf_counter()),
    }
    client.publish(raw_topic, json.dumps(payload), qos=0)


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Virtual sensor publisher for the v11 pipeline")
    parser.add_argument("--dataset", default="data.txt")
    parser.add_argument("--sensors", type=int, default=8)
    parser.add_argument("--discovery-wait", type=float, default=3.0,
                        help="Seconds between discovery windows (pipeline classify/register)")
    parser.add_argument("--registry-timeout", type=float, default=30.0,
                        help="Max seconds to wait for a sensor's registry message")
    parser.add_argument("--op-wait", type=float, default=5.0,
                        help="Seconds to pause after publishing a sensor's block")
    args = parser.parse_args()

    n_sensors = max(1, min(args.sensors, len(ALL_PORTS)))
    ports = ALL_PORTS[:n_sensors]

    print("Loading dataset:", args.dataset)
    cols = load_columns(args.dataset, MAX_POSITION)
    for t, c in cols.items():
        print(f"  {t}: {len(c)} points")

    for p in ports:
        seg = segment_for(cols, p["true_type"], p["position"], NUM_OBSERVATIONS)
        STATE.segments[p["port"]] = seg
        STATE.port_by_label[p["label"]] = p["port"]
        print(f"  {p['label']:22s} {p['true_type']:12s} pos={p['position']:5d} "
              f"discovery={[round(v,3) for v in seg[:WINDOW_SIZE]]}")

    STATE.client = make_client()
    print(f"\nConnected to edge broker {EDGE_BROKER_HOST}:{EDGE_BROKER_PORT}")

    # ---- step 1: retained configuration, read by the pipeline at startup ----
    print("\n=== CONFIG (retained) ===")
    publish_config(STATE.client, ports)
    print("\nConfig published. Start the pipeline now.")
    print("Waiting for the pipeline to come up and start discovery...\n")

    # Time to start the pipeline, which reads the retained configuration
    # and then waits for discovery data on the unknown topic.
    time.sleep(args.discovery_wait)

    # ---- step 2: discovery windows ----
    print("=== DISCOVERY ===")
    for p in ports:
        seg = STATE.segments[p["port"]]
        publish_discovery(STATE.client, p["port"], seg[:WINDOW_SIZE])
        print(f"  [discovery] {p['port']} ({p['label']}): "
              f"{[round(v,3) for v in seg[:WINDOW_SIZE]]}")
        time.sleep(args.discovery_wait)

    # ---- step 3: operation data on the raw topic of each registered sensor ----
    # Wait for the registry message of each sensor (which carries the raw topic),
    # then publish its block.
    print("\n=== OPERATION ===")

    for p in ports:
        port = p["port"]
        # wait until the raw topic is known
        wait_start = time.time()
        raw_topic = None
        while raw_topic is None:
            with STATE.lock:
                raw_topic = STATE.raw_topic_by_port.get(port)
            if raw_topic is None:
                if time.time() - wait_start > args.registry_timeout:
                    print(f"  [operation] {port}: registry not received, skipping")
                    break
                time.sleep(0.2)
        if raw_topic is None:
            continue

        seg = STATE.segments[port]
        publish_operation_block(STATE.client, raw_topic, seg)
        print(f"  [operation] {port} ({p['label']}): published block of "
              f"{len(seg)} pts to {raw_topic}")
        time.sleep(args.op_wait)

    # Time for the pipeline to run the three scenarios over the buffered data.
    time.sleep(2.0)
    STATE.client.loop_stop()
    STATE.client.disconnect()
    print("\nDone. Publisher disconnected.")


if __name__ == "__main__":
    main()