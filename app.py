import gradio as gr
import whisper

# Whisper modelini yükle
model = whisper.load_model("base")  # "small", "tiny", "medium", "large" seçenekleri de kullanılabilir

def transcribe(audio):
    """Ses dosyasını metne çeviren fonksiyon"""
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    
    return result.text

# Gradio arayüzü
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", live=True),  # "source" parametresi kaldırıldı, "live=True" eklendi
    outputs="text",
    title="Whisper Speech-to-Text",
    description="Konuşmayı metne çeviren OpenAI Whisper tabanlı bir uygulama."
)

# Uygulamayı başlat
iface.launch()