import numpy as np
import gradio as gr
from datasets import load_dataset

def generate_random_data():
    # Load the dataset with the `large_random_1k` subset
    dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')
    # All data are stored in the `train` split
    my_1k_data = dataset['train']
    
    random_i = np.random.choice(range(my_1k_data.num_rows))
    
    prompt = my_1k_data['prompt'][random_i]
    image = my_1k_data['image'][random_i]
    seed = my_1k_data['seed'][random_i]
    step = my_1k_data['step'][random_i]
    cfg = my_1k_data['cfg'][random_i]
    sampler = my_1k_data['sampler'][random_i]
    
    return prompt, image, seed, step, cfg, sampler

def random_data():
    prompt, image, seed, step, cfg, sampler = generate_random_data()
    
    data = {
        'Prompt': prompt,
        'Seed': seed,
        'Step': step,
        'CFG': cfg,
        'Sampler': sampler
    }
    
    with open("random_data.txt", "w") as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
    
    return prompt, image, seed, step, cfg, sampler

iface = gr.Interface(fn=random_data, inputs=None, outputs=[
    gr.outputs.Textbox(label="Prompt"),
    gr.outputs.Image(label="Image", type="pil"),
    gr.outputs.Textbox(label="Seed"),
    gr.outputs.Textbox(label="Step"),
    gr.outputs.Textbox(label="CFG"),
    gr.outputs.Textbox(label="Sampler")
], title="Stable Diffusion DB", description="By Falah.G.S AI Developer")

iface.launch(debug=True)
