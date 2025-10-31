import yaml
from audio_io import AudioStream
from vad import VADSegmenter
from asr import ASR
from kws_map import CommandMapper

# Loading the comfig.yml into identifier
def load_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# Set AudioStream attributes to config vals
def main():
    cfg = load_cfg()
    stream = AudioStream(samplerate=cfg["sample_rate"], 
                        block_ms=cfg["block_duration_ms"], 
                        device_index=cfg.get("device_index"))
    
    vad = VADSegmenter(sr=cfg["sample_rate"],
                       th=cfg["vad"]["threshold"],
                       min_speech_ms=cfg["vad"]["min_speech_ms"],
                       max_speech_ms=cfg["vad"]["max_speech_ms"],
                       min_silence_ms=cfg["vad"]["min_silence_ms"])
    asr = ASR(model_size=cfg["model_size"], device=cfg["device"], compute_type=cfg["compute_type"])
    mapper = CommandMapper(cfg["commands"], score_cutoff=80)

    print("Listening for a command: (forward/left/right/back/stop/manual)")
    stream.start()
    try:
        while True:
            chunk = stream.read()
            seg = vad.push(chunk)
            if seg is None: 
                continue
            text = asr.transcribe(seg)
            cmd = mapper.map_text(text)
            print(f"[transcript] {text!r}  ->  [cmd] {cmd}")
            # TODO: publish to ROS2 when later:
            #   rclpy publisher on topic '/wheelchair/voice_cmd' with msg.data = cmd or None
    except KeyboardInterrupt:
        stream.stop()

if __name__ == "__main__":
    main()
