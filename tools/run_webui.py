import gradio as gr
import torch
import traceback
from pathlib import Path

# Import TTSGenerator from the sibling file tts_infer.py (inference.py)
# Ensure the supporting code in tools/webui/inference.py is present
from tools.webui.inference import TTSGenerator

# Global model instance
generator = None

def load_models(model_path, vqgan_checkpoint):
    """Load models and bind to global variables"""
    global generator
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Release old model memory (if exists)
        if generator is not None:
            del generator
            torch.cuda.empty_cache()
            
        generator = TTSGenerator(
            model_path=model_path,
            vqgan_config="modded_dac_vq", # Default vqgan_config
            vqgan_checkpoint=vqgan_checkpoint,
            device=device,
            max_seq_len=8192,
            use_cuda_graph=True
        )
        return gr.update(value="✅ Model loaded successfully!", visible=True)
    except Exception as e:
        error_msg = traceback.format_exc()
        return gr.update(value=f"❌ Loading failed:\n{error_msg}", visible=True)

def generate_audio(prompt_audio, prompt_text, target_text, temperature, top_p, top_k, max_new_tokens):
    """Call the model for inference and audio generation"""
    global generator
    if generator is None:
        raise gr.Error("Please click the button above to load the model first!")
    
    if not prompt_audio:
        raise gr.Error("Please upload prompt audio")
        
    if not target_text:
        raise gr.Error("Please enter target text")

    # When Gradio Audio component is set to type="filepath", prompt_audio is the absolute path string
    with open(prompt_audio, "rb") as f:
        prompt_audio_bytes = f.read()

    try:
        # Call the generate method of the original script
        audio_array, sample_rate = generator.generate(
            text=target_text,
            prompt_texts=[prompt_text] if prompt_text else [],
            prompt_audios=[prompt_audio_bytes],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens
        )
        # Gradio expects the audio return format to be: (sample_rate, numpy_array)
        return (sample_rate, audio_array)
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# ==================== Gradio Interface Setup ====================
with gr.Blocks(title="Fish Speech TTS WebUI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🐟 Fish Speech TTS WebUI")
    gr.Markdown("Zero-shot/Few-shot voice cloning and generation based on Dual-AR model (supports automatic long text slicing)")

    # 1. Model Settings
    with gr.Accordion("⚙️ Model Settings", open=True):
        with gr.Row():
            model_path_input = gr.Textbox(
                label="TTS Model Path", 
                value="./checkpoints/tts-grpo-s2-pro-e394-20260131",
                scale=2
            )
            vqgan_ckpt_input = gr.Textbox(
                label="VQGAN Checkpoint Path", 
                value="./checkpoints/tts-grpo-s2-pro-e394-20260131/step-1380000.pth",
                scale=2
            )
        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(label="Load Status", interactive=False, value="Model not loaded, please click the button on the left.", scale=3)
            
        load_btn.click(fn=load_models, inputs=[model_path_input, vqgan_ckpt_input], outputs=[load_status])

    gr.HTML("<hr>")
    
    # 2. Inference & Generation
    with gr.Row():
        # Left panel: Input parameters
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Prompt Information")
            prompt_audio_input = gr.Audio(
                label="Prompt Audio", 
                type="filepath", 
                value="./test.wav" # Demo audio path
            )
            prompt_text_input = gr.Textbox(
                label="Prompt Text", 
                lines=4, 
                value="In the decade since becoming an electronic ghost, I've often visited the ice plains to see it. It always stays there, silent in the snow, and so I stand silently in the snow, watching it. What made it decide to take out its core and bring light to the underground at a time when the Lahairo civilization had not yet germinated? Tunneler, you too come from there, why... can you choose like this?"
            )
            
            gr.Markdown("### 📝 Target Generation")
            target_text_input = gr.Textbox(
                label="Target Text", 
                lines=5, 
                value="That summer somehow felt so long and so hot. At the time, I just thought it would be fine once summer was over. (sighs) But once summer passed, I could only reminisce. As that summer fades in my memory, I miss it even more."
            )
            
            with gr.Accordion("🔧 Advanced Settings", open=False):
                temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")
                top_k_slider = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Top-k")
                max_new_tokens_slider = gr.Slider(minimum=128, maximum=8192, value=2048, step=128, label="Max New Tokens")
            
            generate_btn = gr.Button("✨ Generate Audio", variant="primary", size="lg")
            
        # Right panel: Output result
        with gr.Column(scale=1):
            gr.Markdown("### 🎧 Output")
            output_audio = gr.Audio(label="Generated Audio")
            
    # Bind click event for the generation button
    generate_btn.click(
        fn=generate_audio,
        inputs=[
            prompt_audio_input, 
            prompt_text_input, 
            target_text_input,
            temperature_slider,
            top_p_slider,
            top_k_slider,
            max_new_tokens_slider
        ],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    # Use 0.0.0.0 to allow LAN/Public access. Note to open port 7860 on the server's firewall.
    app.launch(server_name="0.0.0.0", server_port=7860)