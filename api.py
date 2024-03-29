from io import BytesIO
from typing import Any

import torch
from flask import Flask, Response, request, send_file
from matplotlib import pyplot as plt
from PIL import Image  # type: ignore
from torchvision import transforms  # type: ignore

from generator import generate as gen

app = Flask(__name__)

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# make api
# curl -F "sketch=@/home/jeff/Sketchist/sketch.jpg" http://localhost:5000/generate > out.jpg


@app.after_request
def after_request(response: Any) -> Any:
    header = response.headers
    header["Access-Control-Allow-Origin"] = "*"
    header["Access-Control-Allow-Headers"] = "*"
    return response


@app.route("/generate", methods=["POST"])
def generate() -> Any:
    try:
        img = request.data
    except Exception as err:
        return f"problem with image file: {err}", 400

    img_io = BytesIO()

    # open the image from file -> generate image
    im = Image.open(BytesIO(img)).convert("L")
    im = to_tensor(im)
    im = torch.cat((im, im, im))
    # im = 2 * ((im - im.min()) / (im.max() - im.min())) - 1
    im = (im - 0.5) / 0.5
    out = gen(im)
    out = out * 0.5 +  0.5
    out = torch.squeeze(out)

    # convert to image and write to buffer
    out_img = to_img(out)
    out_img.save(img_io, "JPEG", quality=100)
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")
