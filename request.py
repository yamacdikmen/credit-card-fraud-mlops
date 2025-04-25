import requests
import json

url = "http://127.0.0.1:5001/invocations"

headers = {"Content-Type": "application/json"}

data = {
    "inputs": [[-1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2396, 0.0987,
                0.3638, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704,
                0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669,
                0.1285, -0.1891, 0.1335, -0.0210, 0.0, 149.62]]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print("Prediction:", response.json())
