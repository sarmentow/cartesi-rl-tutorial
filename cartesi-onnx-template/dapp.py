from os import environ
import base64
import numpy as np
import onnxruntime as ort
import logging
import requests

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

rollup_server = environ["ROLLUP_HTTP_SERVER_URL"]
logger.info(f"HTTP rollup_server url is {rollup_server}")

# Load the ONNX model
model_path = "nn.onnx"
session = ort.InferenceSession(model_path)
input_names = [session.get_inputs()[0].name]
output_names = [session.get_outputs()[0].name]
# Get the input and output names
MODEL_DTYPE=np.int64
MODEL_INPUT_SHAPE = (1,)

def str2hex(str):
    """
    Encodes a string as a hex string
    """
    return "0x" + str.encode("utf-8").hex()

def hex2str(hex):
    """
    Decodes a hex string into a regular string
    """
    return bytes.fromhex(hex[2:]).decode("utf-8")

def handle_advance(data):
    logger.info(f"Received advance request data {data}")
    decoded_bytes = base64.b64decode(hex2str(data["payload"]))

    global arr

    arr = np.frombuffer(decoded_bytes, dtype=MODEL_DTYPE).reshape(MODEL_INPUT_SHAPE).tolist()

    outputs = session.run(output_names, {input_names[0]: arr})
    try:
        response = requests.post(
                    rollup_server + "/notice", json={"payload": str2hex(str({"modelOutputs": str(outputs[0][0])}))}
        )
        logger.info(
            f"Received notice status {response.status_code} body {response.content}"
        )
    except Exception as e:
        logger.error(f"Exception while handling advance:{e}")

    return "accept"


def handle_inspect(data):
    logger.info(f"Received inspect request data {data}")
    logger.info("global arr is:", arr)
    return "accept"


handlers = {
    "advance_state": handle_advance,
    "inspect_state": handle_inspect,
}

finish = {"status": "accept"}

while True:
    logger.info("Sending finish")
    response = requests.post(rollup_server + "/finish", json=finish)
    logger.info(f"Received finish status {response.status_code}")
    if response.status_code == 202:
        logger.info("No pending rollup request, trying again")
    else:
        rollup_request = response.json()
        data = rollup_request["data"]
        handler = handlers[rollup_request["request_type"]]
        finish["status"] = handler(rollup_request["data"])
