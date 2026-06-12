import os
import torch
import gradio as gr
from PIL import Image
from torchvision.transforms import transforms

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "Genius-Society/HEp2",
        cache_dir="./__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "Genius-Society/HEp2",
        cache_dir="./__pycache__",
    )

ZH2EN = {
    "上传细胞图像": "Upload a cell picture",
    "状态栏": "Status",
    "图片名": "Picture name",
    "识别结果": "Recognition result",
    "请上传 PNG 格式的 HEp2 细胞图片": "It is recommended to upload HEp2 cell images in PNG format.",
}

TRANSLATE = {
    "Centromere": "着丝粒",
    "Golgi": "高尔基体",
    "Homogeneous": "同质",
    "NuMem": "记忆体",
    "Nucleolar": "核仁",
    "Speckled": "斑核",
}

CLASSES = list(TRANSLATE.keys())

I18N = gr.I18n(
    zh={key: key for key in ZH2EN}
    | {TRANSLATE[key]: TRANSLATE[key] for key in TRANSLATE},
    en=ZH2EN | {TRANSLATE[key]: key for key in TRANSLATE},
)


def embeding(img_path: str):
    compose = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomAffine(5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return compose(img)


def infer(target: str):
    status = "Success"
    filename = result = None
    try:
        model = torch.load(
            f"{MODEL_DIR}/save.pt",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if not target:
            raise ValueError("请上传细胞图片")

        torch.cuda.empty_cache()
        input: torch.Tensor = embeding(target)
        output: torch.Tensor = model(input.unsqueeze(0))
        predict = torch.max(output.data, 1)[1]
        filename = os.path.basename(target)
        result = I18N(TRANSLATE[CLASSES[predict]])

    except Exception as e:
        status = f"{e}"

    return status, filename, result


if __name__ == "__main__":
    example_imgs = []
    for cls in CLASSES:
        example_imgs.append(f"{MODEL_DIR}/examples/{cls}.png")

    gr.Interface(
        fn=infer,
        inputs=gr.Image(type="filepath", label=I18N("上传细胞图像")),
        outputs=[
            gr.Textbox(label=I18N("状态栏"), buttons=["copy"]),
            gr.Textbox(label=I18N("图片名"), buttons=["copy"]),
            gr.Textbox(label=I18N("识别结果"), buttons=["copy"]),
        ],
        title=I18N("请上传 PNG 格式的 HEp2 细胞图片"),
        examples=example_imgs,
        flagging_mode="never",
        cache_examples=False,
    ).launch(
        theme=gr.themes.Ocean(),
        css="#gradio-share-link-button-0 { display: none; }",
        ssr_mode=False,
        i18n=I18N,
    )
