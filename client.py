import gymnasium as gym
import numpy as np
from os import system

env = gym.make("FrozenLake-v1", render_mode="human")
observation, info = env.reset()
print(observation)


import base64
import requests
import json

# Function to decode hex string to a regular string
def hex2str(hex_str):
    """
    Decodes a hex string into a regular string
    """
    return bytes.fromhex(hex_str[2:]).decode("utf-8")

# The GraphQL query
query = """
query allNotices {
  notices {
    edges {
      node {
        payload
      }
    }
  }
}
"""


for _ in range(1000):
    # Ideally you'd interact with the machine through the smart contract interface, but for the proof of concept here we just use the cartesi cli
    command = system(f"cartesi send generic --chain-id=31337 --rpc-url=http://127.0.0.1:8545 --mnemonic-passphrase=\"test test test test test test test test test test test junk\" --dapp=0xab7528bb862fb57e8a2bcd567a2e929a0be56a5e --input-encoding=string --input=\"{base64.b64encode(np.array(observation).tobytes()).decode('utf-8')}\"")
    # Send the GraphQL request
    url = "http://localhost:8080/graphql"
    response = requests.post(url, json={'query': query})

    # Parse the response JSON
    data = response.json()

    # Extract the last payload from the edges array
    edges = data['data']['notices']['edges']
    if edges:
        last_payload = edges[-1]['node']['payload']
        
        # Decode the payload using hex2str
        decoded_payload = hex2str(last_payload)
        decoded_payload = decoded_payload.replace("\'", "\"")

        # Load the decoded payload as a JSON object
        payload_json = json.loads(decoded_payload)
        action = int(payload_json['modelOutputs'])
        # Print or return the result
        print(payload_json)
    else:
        print("No edges found in the response.")
        break
    print("Action:", action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
