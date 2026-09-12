import requests
import json

with open('data/raw/pump/id_00/normal/00000000.wav', 'rb') as f:
    files = {'audio': ('00000000.wav', f, 'audio/wav')}
    data = {'machine_id': '00'}
    res = requests.post('http://127.0.0.1:5000/api/upload_audio', files=files, data=data)
    d = res.json()
    result = d.get("result", {})
    if not result:
        print("ERROR:", d)
    else:
        print("Decision:", result.get("decision"))
        print("Prediction:", result.get("prediction"))
        print("Confidence:", result.get("confidence"))
        print("OOD Status:", result.get("ood_status"))
