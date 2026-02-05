import gradio as gr
import requests
import base64
import json
from app.main import app
import uvicorn
import threading
import time

# Start FastAPI server in background
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI in a separate thread
threading.Thread(target=start_fastapi, daemon=True).start()
time.sleep(3)  # Wait for server to start

def analyze_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file"
    
    try:
        # Read and encode audio
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Call local FastAPI
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"audio_base64": audio_b64}
        )
        
        if response.status_code == 200:
            result = response.json()
            return f"""
## Analysis Result

**Classification:** {result['classification']}
**Confidence:** {result['confidence']:.2%}
**Language:** {result['language']}
**Processing Time:** {result['processing_time_ms']}ms

### Explanation
{json.dumps(result['explanation'], indent=2)}
"""
        else:
            return f"Error: {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Markdown(label="Analysis Result"),
    title="🎙️ EchoTruth AI - Voice Authenticity Detection",
    description="Upload an audio file to detect if it's AI-generated or human voice",
    examples=None
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
