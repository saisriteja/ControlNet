from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
from diffusers.utils import load_image








def inference(img_path, pipe):

    image_inp = load_image(img_path)

    image_inp = np.array(image_inp)
    cv2.imwrite("image_inp.jpg", image_inp)

    image_inp = image_inp / 255.0
    image_inp = (image_inp - 0.5) / 0.5

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid


    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(1)]
    prompt = "Remove the unwanted flare in this image."
    output = pipe(
        [prompt] * 1,
        [image_inp] * 1,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 1,
        generator=generator,
        num_inference_steps=50,
    )
    grid = image_grid(output.images, 1, 1)

    # save the image grid
    grid.save("output.png")




if __name__ == '__main__':
    model_output_path = "/home/cilab/teja/diffusers/flaremodel_v1"
    image_inp_path = "/media/cilab/data/NTIRE/flare/Flare7K/images/train/conditional/20240205_221507_16.jpg"
    controlnet = ControlNetModel.from_pretrained(model_output_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    # noise scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    inference(image_inp_path, pipe)

