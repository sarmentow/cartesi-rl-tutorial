<!-- TODO: check if it's possible to run the dapp from the tutorial without editing the composer-validator yaml -->
<!-- TODO: add cover gif for application -->
# Running machine learning models in Cartesi machines
Pedro Lins, João Pedro Belga, Leonardo Camargo

## Introduction

Artificial intelligence is a powerful set of technologies with a multitude of potential applications in web3. Unfortunately, due to high computational requirements, most machine learning algorithms have been out of reach for web3 developers. 

Thankfully, Cartesi rollups make it technically feasible to run machine learning algorithms with all of the guarantees that blockchain consensus is usually associated with. 

This post is a tutorial that goes from training a model, deploying it in a rollup, and writing an application to send inputs to the model and collect its outputs.

By the end of this tutorial you'll have a non-trivial RL model running inference inside a Cartesi machine!

## Tutorial

### Step 1: Training and exporting your model

For this tutorial, we'll be training a reinforcement learning model from scratch, using the Stable Baselines3 library, to cross a frozen lake without falling into any holes - i.e. the [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment from [Gymnasium](https://gymnasium.farama.org/).

Use the [Google Colab notebook](https://colab.research.google.com/drive/1edMqdrb3Glgf7w-_5bamZigWv06LFMwR?usp=sharing) to train your model from scratch.

### Step 2: Setting up the dapp

Now you'll set up your dapp. There are 2 ways of doing this, one of them we have done most of the work for you already: 

1. [Recommended] We have provided a dapp template which comes with all you need to run inference inside a Cartesi rollup. Simply clone the repo:

    ```
    git clone https://github.com/sarmentow/cartesi-onnx-template.git
    ```

    The template comes with a Dockerfile which installs a pre-built wheel for the `onnxruntime` package, and a dapp.py file which puts the inputs sent to the rollup into the model, and emits the model's outputs as notices that you can query through the GraphQL endpoint.

    You should replace the `simple_nn.onnx` file inside it with the one you've created in step 1 (make sure to rename it to `simple_nn.onnx` or change the model filename in `dapp.py`). Make sure you have the model parameters set appropriately in your `dapp.py`. For our example:
    ```python
    MODEL_DTYPE=np.int64
    MODEL_INPUT_SHAPE = (1,)
    ```

    <!-- Also, make sure to run the `config-cli.sh` file, you only have to do this once after you've installed the Cartesi CLI.
    ```
    bash config-cli.sh
    ``` -->


2. You can also start your project from scratch. First you'll need to build the `onnxruntime` library for RISC-V, and write your own dapp logic according to your project's needs. [We have a tutorial on how to build the `onnxruntime`for RISC-V from source](https://gist.github.com/sarmentow/66bc6a03f0906c1b3c23dab861f06830) and have also made available a [pre-built wheel for onnxruntime](https://github.com/sarmentow/riscv-wheels). 

### Step 3: Creating the client application
First, install the dependencies:
```
pip install gymnasium onnxruntime numpy
```

Here's an example client application you can just copy and paste to send observations to our model running in the Cartesi machine and apply the action:

```python
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
    data = response.json()

    # Extract the last payload from the edges array
    edges = data['data']['notices']['edges']
    if edges:
        last_payload = edges[-1]['node']['payload']
        decoded_payload = hex2str(last_payload)
        decoded_payload = decoded_payload.replace("\'", "\"")
        payload_json = json.loads(decoded_payload)
        action = int(payload_json['modelOutputs'])
    else:
        print("No edges found in the response.")
        break
    print("Action:", action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

With the client script and dapp in place you can run `cartesi build && cartesi run` from inside the dapp template on one terminal, and run the client script from another terminal session `python3 client.py`.

If you want to make changes to the dapp, we recommend using Nonodo to be able to quickly see your changes take effect. You can use Nonodo as so:
`nonodo -- python3 dapp.py`

## Congrats!
You have just trained an RL model, put it inside a Cartesi rollup, and wrote an application which leverages the rollup to perform completely verifiable and deterministic inference for controlling a character in a simulated environment!

Our model isn't great... Can you make it better?

This is just the beginning in your journey to writing AI-powered web3 applications.  

## Use cases for machine learning
> The biggest challenge for the evolution of Web3-AI might be overcoming its own reality distortion field

[Article from CoinDesk](https://www.coindesk.com/opinion/2024/07/16/web3-ai-whats-real-and-whats-hype/)

This tutorial is a basic demonstration of the workflow for writing AI applications for Cartesi rollups. Technically speaking, you've just implemented a proof-of-inference application. 

In terms of real word applications... The singular characteristics of blockcahin applications, and the requirements that rollups have in order to guarantee determinism (and verifiability) of your program (plus, how recent this technology is) make this an unknown terrain ripe for innovation. It's up for hackers to build the applications that will define the future of web3 - and perhaps AI models have a place in it.

## Acknowledgements
This work was done as part of the Cartesi Seed Grants program. We'd like to thank the Cartesi DevAd team, and the community for their feedback and support in doing this work.


