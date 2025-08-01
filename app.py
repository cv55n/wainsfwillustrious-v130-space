import spaces
import gradio as gr
import numpy as np
import PIL.Image
from PIL import Image
import random
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch
from compel import Compel, ReturnedEmbeddingsType
# import os
# from gradio_client import Client

# client = Client("dhead/ntr-mix-illustrious-xl-noob-xl-xiii-sdxl", hf_token=os.getenv("HUGGING_FACE_TOKEN"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# certifique-se de usar torch.float16 consistentemente em todo o pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "dhead/wai-nsfw-illustrious-sdxl-v140-sdxl",
    torch_dtype=torch.float16,
    variant="fp16", # usar explicitamente a variante fp16
    use_safetensors=True # usar safetensors se disponível
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# força todos os componentes a usar o mesmo dtype
pipe.text_encoder.to(torch.float16)
pipe.text_encoder_2.to(torch.float16)
pipe.vae.to(torch.float16)
pipe.unet.to(torch.float16)

# inicializa o compel para processamento de prompt longo
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    truncate_long_prompts=False
)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1216

# função de processamento de prompt longo simples
def process_long_prompt(prompt, negative_prompt=""):
    """Simple long prompt processing using Compel"""

    try:
        conditioning, pooled = compel([prompt, negative_prompt])
        
        return conditioning, pooled
    except Exception as e:
        print(f"falha no processamento do prompt longo: {e}, retornando ao processamento padrão")
        
        return None, None

@spaces.GPU
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    # remove o aviso de limite de 60 palavras e adicione uma verificação de prompt longa
    use_long_prompt = len(prompt.split()) > 60 or len(prompt) > 300
        
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        # tenta primeiro o processamento de prompt longo se o prompt for longo
        if use_long_prompt:
            print("utilizando processamento de prompt longo...")

            conditioning, pooled = process_long_prompt(prompt, negative_prompt)
            
            if conditioning is not None:
                output_image = pipe(
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]

                return output_image
        
        # retorna ao processamento padrão
        output_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        return output_image
    except RuntimeError as e:
        print(f"erro durante a geração: {e}")

        # retorna uma imagem em branco com mensagem de erro
        error_img = Image.new('RGB', (width, height), color=(0, 0, 0))
        
        return error_img
    
css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        with gr.Row():
            prompt = gr.Text(
                label="prompt",
                show_label=False,
                max_lines=1,
                placeholder="insira seu prompt (prompts longos são suportados automaticamente)",
                container=False
            )

            run_button = gr.Button("rodar", scale=0)

        result = gr.Image(format="png", label="resultado", show_label=False)
        
        with gr.Accordion("configurações avançadas", open=False):

            negative_prompt = gr.Text(
                label="prompt negativo",
                max_lines=1,
                placeholder="insira um prompt negativo",
                value="monochrome, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn,"
            )

            seed = gr.Slider(
                label="seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0
            )

            randomize_seed = gr.Checkbox(label="aleatorizar seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="largura",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024
                )

                height = gr.Slider(
                    label="altura",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=MAX_IMAGE_SIZE
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="escala de guia",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=7
                )

                num_inference_steps = gr.Slider(
                    label="número de etapas de inferência",
                    minimum=1,
                    maximum=28,
                    step=1,
                    value=28
                )

    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result]
    )

demo.queue().launch()