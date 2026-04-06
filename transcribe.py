import sys
import io
from faster_whisper import WhisperModel

# 1. WINDOWS ENCODING FIX: Tamil/Tanglish characters terminal-la crash aagama irukka
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def transcribe(audio_path):
    try:
        # 'small' model CPU-la run aaga konjam neram edukum
        # RAM kuraivaga iruntha 'base' model try pannalam
        model = WhisperModel("small", device="cpu", compute_type="int8")
        
        # language="ta" nu kudutha, whisper Tamil-la thaan output tharum
        # Tanglish-ku language specify pannama iruntha automatic-ah detect pannikum
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        # Text-ah join pannum pothu strip() panni extra spaces-ah remove panrom
        text = " ".join([segment.text.strip() for segment in segments])
        
        # 2. OUTPUT FIX: Node.js-ku sariyaana output-ah flush panrom
        print(text, flush=True)
        
    except Exception as e:
        sys.stderr.write(f"Whisper Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe(sys.argv[1])
    else:
        print("Error: No audio path provided")