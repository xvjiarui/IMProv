import torch
from enum import Enum

class PromptMode(Enum):
    H_TL = "horizontal_top_left"
    H_TR = "horizontal_top_right"
    H_BL = "horizontal_bottom_left"
    H_BR = "horizontal_bottom_right"
    V_TL = "vertical_top_left"
    V_TR = "vertical_top_right"
    V_BL = "vertical_bottom_left"
    V_BR = "vertical_bottom_right"


class PromptMode2Image(Enum):
    H_L = "horizontal_left"
    H_R = "horizontal_right"
    V_T = "vertical_top"
    V_B = "vertical_bottom"

def create_grid_from_batch(
    batch, layout=None, padding=1, border="white", grid_2image=False, return_prompt=False
):
    if grid_2image:
        if layout is None:
            layout = PromptMode2Image.H_R
        canvas, mask, prompt = create_grid2image_from_batch(
            batch, layout=layout, padding=padding, border=border
        )
    else:
        if layout is None:
            layout = PromptMode.H_BR
        canvas, mask, prompt = create_grid4image_from_batch(
            batch, layout=layout, padding=padding, border=border
        )

    if return_prompt:
        return canvas, mask, prompt
    else:
        return canvas, mask

def create_grid4image_from_batch(batch, layout=PromptMode.H_BR, padding=1, border="white", zero_out_query_mask=True):
    support_img = batch["support_img"]
    support_mask = batch["support_mask"]

    query_img = batch["query_img"]
    if zero_out_query_mask:
        query_mask = torch.zeros_like(batch["query_img"])
    else:
        query_mask = batch["query_mask"]

    h, w = support_img.shape[-2:]
    assert support_mask.shape[-2:] == (h, w), f"{support_mask.shape[-2:]} != {(h, w)}"
    assert query_img.shape[-2:] == (h, w), f"{query_img.shape[-2:]} != {(h, w)}"

    canvas = torch.ones(
        (
            *support_img.shape[:-2],
            2 * h + 2 * padding,
            2 * w + 2 * padding,
        )
    )

    if border == "black":
        canvas *= 0

    if layout == PromptMode.H_BR:

        """
        support_img | support_mask
        --------------------------
        query_img   | query_mask
        """

        # TL
        canvas[..., :h, :w] = support_img
        # BL
        canvas[..., -h:, :w] = query_img
        # TR
        canvas[..., :h, -w:] = support_mask
        # BR
        canvas[..., -h:, -w:] = query_mask

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., -h:, -w:] = 1

        prompt = "Left - input image, right - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.H_BL:

        """
        support_mask | support_img
        --------------------------
        query_mask   | query_img
        """

        # TL
        canvas[..., :h, :w] = support_mask
        # BL
        canvas[..., -h:, :w] = query_mask
        # TR
        canvas[..., :h, -w:] = support_img
        # BR
        canvas[..., -h:, -w:] = query_img

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., -h:, :w] = 1

        prompt = "Right - input image, left - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.H_TR:

        """
        query_img   | query_mask
        --------------------------
        support_img | support_mask
        """

        # TL
        canvas[..., :h, :w] = query_img
        # BL
        canvas[..., -h:, :w] = support_img
        # TR
        canvas[..., :h, -w:] = query_mask
        # BR
        canvas[..., -h:, -w:] = support_mask

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., :h, -w:] = 1

        prompt = "Left - input image, right - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.H_TL:

        """
        query_mask   | query_img
        --------------------------
        support_mask | support_img
        """

        # TL
        canvas[..., :h, :w] = query_mask
        # BL
        canvas[..., -h:, :w] = support_mask
        # TR
        canvas[..., :h, -w:] = query_img
        # BR
        canvas[..., -h:, -w:] = support_img

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., :h, :w] = 1

        prompt = "Right - input image, left - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.V_BR:

        """
        support_img | query_img
        -----------------------
        support_mask| query_mask
        """

        # TL
        canvas[..., :h, :w] = support_img
        # BL
        canvas[..., -h:, :w] = support_mask
        # TR
        canvas[..., :h, -w:] = query_img
        # BR
        canvas[..., -h:, -w:] = query_mask

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., -h:, -w:] = 1

        prompt = "Top - input image, bottom - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.V_BL:

        """
        query_img   | support_img
        --------------------------
        query_mask  | support_mask
        """

        # TL
        canvas[..., :h, :w] = query_img
        # BL
        canvas[..., -h:, :w] = query_mask
        # TR
        canvas[..., :h, -w:] = support_img
        # BR
        canvas[..., -h:, -w:] = support_mask

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., -h:, :w] = 1

        prompt = "Top - input image, bottom - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.V_TR:

        """
        support_mask | query_mask
        --------------------------
        support_img  | query_img
        """

        # TL
        canvas[..., :h, :w] = support_mask
        # BL
        canvas[..., -h:, :w] = support_img
        # TR
        canvas[..., :h, -w:] = query_mask
        # BR
        canvas[..., -h:, -w:] = query_img

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., :h, -w:] = 1

        prompt = "Bottom - input image, top - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode.V_TL:

        """
        query_mask   | support_mask
        ----------------------------
        query_img    | support_img
        """

        # TL
        canvas[..., :h, :w] = query_mask
        # BL
        canvas[..., -h:, :w] = query_img
        # TR
        canvas[..., :h, -w:] = support_mask
        # BR
        canvas[..., -h:, -w:] = support_img

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # BR
        mask[..., :h, :w] = 1

        prompt = "Bottom - input image, top - Black and white foreground "
        "background segmentation of {class_name}"

    else:
        raise ValueError(f"Unknown layout {layout}")

    return canvas, mask, prompt

def create_grid2image_from_batch(batch, layout=PromptMode2Image.H_R, padding=1, border="white"):
    query_img = batch["query_img"]
    # query_mask = batch["query_mask"]
    query_mask = torch.zeros_like(batch["query_img"])

    h, w = query_img.shape[-2:]
    assert (
        query_img.shape[-2:] == query_mask.shape[-2:]
    ), f"{query_img.shape[-2:]} != {query_mask.shape[-2:]}"

    canvas = torch.ones(
        (
            *query_img.shape[:-2],
            2 * h + 2 * padding,
            2 * w + 2 * padding,
        )
    )

    if border == "black":
        canvas *= 0

    if layout == PromptMode2Image.V_B:

        """
        query_img
        ----------
        query_mask
        """

        # T
        canvas[..., :h, int(0.5 * w) : int(1.5 * w)] = query_img
        # B
        canvas[..., -h:, int(0.5 * w) : int(1.5 * w)] = query_mask

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # B
        mask[..., -h:, int(0.5 * w) : int(1.5 * w)] = 1

        prompt = "Top - input image, bottom - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode2Image.V_T:

        """
        query_mask
        ----------
        query_img
        """

        # T
        canvas[..., :h, int(0.5 * w) : int(1.5 * w)] = query_mask
        # B
        canvas[..., -h:, int(0.5 * w) : int(1.5 * w)] = query_img

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # T
        mask[..., :h, int(0.5 * w) : int(1.5 * w)] = 1

        prompt = "Bottom - input image, top - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode2Image.H_R:
        """
        query_img   | query_mask
        """

        # L
        canvas[..., int(0.5 * h) : int(1.5 * h), :w] = query_img
        # R
        canvas[..., int(0.5 * h) : int(1.5 * h), -w:] = query_mask

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # R
        mask[..., int(0.5 * h) : int(1.5 * h), -w:] = 1

        prompt = "Left - input image, right - Black and white foreground "
        "background segmentation of {class_name}"

    elif layout == PromptMode2Image.H_L:
        """
        query_mask   | query_img
        """

        # L
        canvas[..., int(0.5 * h) : int(1.5 * h), :w] = query_mask
        # R
        canvas[..., int(0.5 * h) : int(1.5 * h), -w:] = query_img

        mask = torch.zeros((canvas.shape[0], *canvas.shape[-2:]))
        # L
        mask[..., int(0.5 * h) : int(1.5 * h), :w] = 1

        prompt = "Right - input image, left - Black and white foreground "
        "background segmentation of {class_name}"

    else:
        raise ValueError(f"Unknown layout {layout}")

    return canvas, mask, prompt