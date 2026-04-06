import sys
from faster_whisper import WhisperModel

def transcribe(audio_path):
    # 'small' model 16GB RAM-la super fast-ah odum
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5, language="ta")
    
    text = "".join([segment.text for segment in segments])
    print(text) # Intha text thaan Node.js-ku pogum

if __name__ == "__main__":
    transcribe(sys.argv[1])