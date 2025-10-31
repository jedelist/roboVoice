# roboVoice

## 1. For computer/laptop running:

### Set up virtual environment:
```bash
python3 -m venv venv && source venv/bin/activate
pip install faster-whisper torch torchaudio sounddevice numpy scipy rapidfuzz pyyaml
pip install silero-vad==5.1 

#OR 
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Set device:

Run `python3 list_devices.py` and find the index of the microphone connected to computer.

In the `config.yml`, set device_index variable to the number of the microphone from running the above line.

```bash
# ...
device_index: 2
# ...
``` 

### Run Inference in Real Time:

```bash
python main_realtime_asr.py
```