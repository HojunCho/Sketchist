from io import BytesIO
from typing import Any

import torch
from flask import Flask, Response, send_file
from PIL import Image  # type: ignore
from torchvision import transforms  # type: ignore

from generator import generate as gen

app = Flask(__name__)

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


# command to run the API with a hot reload on save
# FLASK_APP=./api.py FLASK_DEBUG=1 flask run


@app.route("/generate")
def generate() -> Any:
    img_io = BytesIO()

    # open the image from file -> generate image
    img = to_tensor(Image.open("./sketch.jpg"))
    out = gen(img)
    out = torch.squeeze(out)

    # convert to image and write to buffer
    out_img = to_img(out)
    out_img.save(img_io, "JPEG", quality=100)
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")
