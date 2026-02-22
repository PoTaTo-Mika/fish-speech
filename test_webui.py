import gradio as gr
import torch
import traceback
from pathlib import Path

# 从你提供的同级文件 tts_infer.py 中导入 TTSGenerator
# 确保你的原代码文件名为 tts_infer.py
from tts_infer import TTSGenerator

# 全局模型实例
generator = None

def load_models(model_path, vqgan_checkpoint):
    """加载模型并绑定到全局变量"""
    global generator
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 释放旧模型显存（如果存在）
        if generator is not None:
            del generator
            torch.cuda.empty_cache()
            
        generator = TTSGenerator(
            model_path=model_path,
            vqgan_config="modded_dac_vq", # 默认的 vqgan_config
            vqgan_checkpoint=vqgan_checkpoint,
            device=device,
            max_seq_len=8192,
            use_cuda_graph=True
        )
        return gr.update(value="✅ 模型加载成功！", visible=True)
    except Exception as e:
        error_msg = traceback.format_exc()
        return gr.update(value=f"❌ 加载失败:\n{error_msg}", visible=True)

def generate_audio(prompt_audio, prompt_text, target_text, temperature, top_p, top_k, max_new_tokens):
    """调用模型进行推理生成音频"""
    global generator
    if generator is None:
        raise gr.Error("请先点击上方按钮加载模型！")
    
    if not prompt_audio:
        raise gr.Error("请上传参考音频 (Prompt Audio)")
        
    if not target_text:
        raise gr.Error("请输入目标文本 (Target Text)")

    # Gradio Audio 组件设置为 type="filepath" 时，prompt_audio 为文件的绝对路径字符串
    with open(prompt_audio, "rb") as f:
        prompt_audio_bytes = f.read()

    try:
        # 调用原始脚本的 generate 方法
        audio_array, sample_rate = generator.generate(
            text=target_text,
            prompt_texts=[prompt_text] if prompt_text else [],
            prompt_audios=[prompt_audio_bytes],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens
        )
        # Gradio 期望的音频返回格式为: (sample_rate, numpy_array)
        return (sample_rate, audio_array)
    except Exception as e:
        raise gr.Error(f"生成失败: {str(e)}")

# ==================== Gradio 界面搭建 ====================
with gr.Blocks(title="Fish Speech TTS WebUI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🐟 Fish Speech TTS WebUI")
    gr.Markdown("基于 Dual-AR 模型的零样本/少样本语音克隆与生成 (支持长文本自动切片)")

    # 1. 模型加载区
    with gr.Accordion("⚙️ 模型设置 (Model Settings)", open=True):
        with gr.Row():
            model_path_input = gr.Textbox(
                label="TTS 模型路径 (Model Path)", 
                value="./checkpoints/tts-grpo-s2-pro-e394-20260131",
                scale=2
            )
            vqgan_ckpt_input = gr.Textbox(
                label="VQGAN 检查点路径 (VQGAN Checkpoint)", 
                value="./checkpoints/tts-grpo-s2-pro-e394-20260131/step-1380000.pth",
                scale=2
            )
        with gr.Row():
            load_btn = gr.Button("🚀 加载模型 (Load Model)", variant="primary", scale=1)
            load_status = gr.Textbox(label="加载状态", interactive=False, value="尚未加载模型，请点击左侧按钮。", scale=3)
            
        load_btn.click(fn=load_models, inputs=[model_path_input, vqgan_ckpt_input], outputs=[load_status])

    gr.HTML("<hr>")
    
    # 2. 推理生成区
    with gr.Row():
        # 左侧面板：输入参数
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 参考信息 (Prompt)")
            prompt_audio_input = gr.Audio(
                label="参考音频 (Prompt Audio)", 
                type="filepath", 
                value="./test.wav" # 将您提供的音频作为默认路径（如果页面上找不到此文件，Gradio 会留空等待用户上传）
            )
            prompt_text_input = gr.Textbox(
                label="参考文本 (Prompt Text)", 
                lines=4, 
                value="在变成电子幽灵后的十多年里，我经常前往冰原看它。它总是静默无言地停驻在雪中，于是我也静静站在雪中望着它。 是什么让它在拉海洛文明尚未萌发的时刻，决定取出炉芯，将光带给地下呢？隧者，明明你也来自那里，你为何……能够这样选择。"
            )
            
            gr.Markdown("### 📝 目标生成 (Target)")
            target_text_input = gr.Textbox(
                label="需要合成的文本 (Target Text)", 
                lines=5, 
                value="那一年夏天不知为何时间那么长，那么热，当时只想着夏天过去就好了，(叹气)但 夏天过去了，我只能回忆，当那个夏天在我的记忆里越来越淡，我便对它多一分想念。"
            )
            
            with gr.Accordion("🔧 高级生成参数 (Advanced Settings)", open=False):
                temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")
                top_k_slider = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Top-k")
                max_new_tokens_slider = gr.Slider(minimum=128, maximum=8192, value=2048, step=128, label="Max New Tokens")
            
            generate_btn = gr.Button("✨ 生成音频 (Generate)", variant="primary", size="lg")
            
        # 右侧面板：输出结果
        with gr.Column(scale=1):
            gr.Markdown("### 🎧 输出结果 (Output)")
            output_audio = gr.Audio(label="生成的音频 (Generated Audio)")
            
    # 绑定生成按钮的点击事件
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
    # 使用 0.0.0.0 允许局域网/公网访问。GCP 服务器上注意配置防火墙开放 7860 端口。
    app.launch(server_name="0.0.0.0", server_port=7860)