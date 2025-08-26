import requests

url = "https://semanticanalysis.sliplane.app/"
data = {"text": "Your Comment Here"}

response = requests.post(url, json=data)
if response.status_code == 200:
    result = response.json()
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']}")
else:
    print("Error:", response.json())
