import requests

url = "http://localhost:8000/process-audio"

with open("data/my_noisy_audio.wav", "rb") as f:

    response = requests.post(
        url,
        files={
            "file": (
                "my_noisy_audio.wav",
                f,
                "audio/wav"
            )
        }
    )

print(response.json())