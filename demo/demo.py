import argparse
import numpy as np
import os
from contextlib import ExitStack
import torch
import torchvision.transforms as T
from PIL import Image

from improv.pipelines.pipeline_improv import  IMProvPipeline
from improv.modeling.meta_arch.utils import create_grid_from_batch


# for pascal mask
def extract_ignore_idx(mask):
    mask = np.array(mask)
    mask[mask == 255] = 0
    mask[mask > 0] = 255
    return Image.fromarray(mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A demo script.")
    parser.add_argument(
        "--hf-dir",
        type=str,
        default="xvjiarui/IMProv-v1-0",
        help="path to HF pipeline directory",
    )
    parser.add_argument(
        "--support-input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "examples", "2010_002030.jpg"),
        # default=os.path.join(os.path.dirname(__file__), "examples", "2008_004630.jpg"),
        help="path to support input",
    )
    parser.add_argument(
        "--support-mask",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "examples", "2010_002030.png"),
        # default=os.path.join(os.path.dirname(__file__), "examples", "2008_004630.png"),
        help="path to support mask",
    )
    parser.add_argument(
        "--query-input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "examples", "2007_000033.jpg"),
        help="path to query input",
    )
    parser.add_argument(
        "--query-mask",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "examples", "2007_000033.png"),
        help="path to query mask",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Left - input image, right - Black and white foreground background segmentation of airplane.",
        help="text prompt",
    )
    parser.add_argument("--output", type=str, help="path to output")
    args = parser.parse_args()

    hf_dir = args.hf_dir
    print(f"Loading model from {hf_dir}")
    pipeline = IMProvPipeline.from_pretrained(pretrained_model_name_or_path=hf_dir)
    pipeline = pipeline.to("cuda")
    print(f"Prompt: {args.prompt}")

    support_img = Image.open(args.support_input).convert("RGB")
    # support_mask = Image.open(args.support_mask).convert("RGB")
    # NOTE: comment this line out for other images
    support_mask = Image.open(args.support_mask)
    support_mask = extract_ignore_idx(support_mask)
    query_img = Image.open(args.query_input).convert("RGB")

    image_size = 224
    padding = 1
    image_transform = T.Compose(
        [
            T.Resize(
                (image_size // 2 - padding, image_size // 2 - padding),
                T.InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
        ]
    )

    support_img = image_transform(support_img)
    support_mask = image_transform(support_mask)
    query_img = image_transform(query_img)

    batch = {
        "support_img": support_img.unsqueeze(0),
        "support_mask": support_mask.unsqueeze(0),
        "query_img": query_img.unsqueeze(0),
    }
    init_image, input_mask = create_grid_from_batch(batch)

    init_image = init_image.to(pipeline.device)
    input_mask = input_mask.to(pipeline.device)
    batch_size = init_image.shape[0]
    input_prompts = [args.prompt] * batch_size

    query_height, query_width = batch["query_img"].shape[-2:]

    generator = torch.Generator(device=pipeline.device).manual_seed(42)
    with ExitStack() as stack:
        stack.enter_context(torch.no_grad())

        raw_inpaint = pipeline(
            input_prompts,
            image=init_image,
            mask_image=input_mask,
            generator=generator,
            height=init_image.shape[-2],
            width=init_image.shape[-1],
            guidance_scale=1.0,
            num_inference_steps=1,
            choice_temperature=0.0,
            output_type="torch",
        ).images

    # TODO: check without blending
    inpainted_images = raw_inpaint * input_mask.unsqueeze(1) + init_image * (
        1 - input_mask.unsqueeze(1)
    )
    inpainted_images = inpainted_images.detach().cpu()

    inpainted_images = [T.ToPILImage("RGB")(img) for img in inpainted_images]
    # outputs = torch.stack(
    #     [T.ToTensor()(img)[..., -query_height:, -query_width:] for img in inpainted_images]
    # )

    inpainted_images[0].save(args.output)
