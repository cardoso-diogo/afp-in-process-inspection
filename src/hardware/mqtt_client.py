# src/hardware/mqtt_client.py

import paho.mqtt.client as mqtt
import json


class MQTTClient:
    """A client for handling MQTT communication for the scanner control project."""

    def __init__(self, broker, port, client_id):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.broker = broker
        self.port = port

        # These will be set by the main application to link events to actions
        self.on_control_message = None
        self.on_robot_message = None

        # Topics will be set during connection
        self.control_topic = None
        self.robot_topic = None

    def connect(self, control_topic, robot_topic):
        """Connects to the broker and starts the network loop."""
        print("MQTT Client: Attempting to connect...")
        self.control_topic = control_topic
        self.robot_topic = robot_topic

        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"MQTT Client: Connection failed - {e}")
            raise

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"MQTT Client: Connected to broker. Subscribing to topics.")
            client.subscribe(self.control_topic)
            client.subscribe(self.robot_topic)
        else:
            print(f"MQTT Client: Failed to connect, return code {rc}")

    def _on_message(self, client, userdata, msg):
        """Parses incoming messages and triggers the appropriate callback."""
        payload_str = msg.payload.decode()

        if msg.topic == self.control_topic:
            if self.on_control_message:
                try:
                    data = json.loads(payload_str)
                    if "value" in data and isinstance(data["value"], bool):
                        self.on_control_message(data["value"])
                except json.JSONDecodeError:
                    pass  # Ignore malformed control messages

        elif msg.topic == self.robot_topic:
            if self.on_robot_message:
                try:
                    data = json.loads(payload_str)
                    timestamp = data.get("timestamp")
                    value_dict = data.get("value")
                    if timestamp is not None and value_dict is not None:
                        self.on_robot_message(timestamp, value_dict)
                except json.JSONDecodeError:
                    pass  # Ignore malformed robot messages

    def disconnect(self):
        """Stops the network loop and disconnects."""
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT Client: Disconnected.")