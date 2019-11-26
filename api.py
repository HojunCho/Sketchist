from io import BytesIO
from typing import Any

import torch
from flask import Flask, Response, request, send_file
from PIL import Image  # type: ignore
from torchvision import transforms  # type: ignore

from generator import generate as gen

app = Flask(__name__)

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


# command to run the API with a hot reload on save. curl command sends a post request with the file to the API endpoint
# FLASK_APP=./api.py FLASK_DEBUG=1 flask run
# curl -F "sketch=@/home/jeff/Sketchist/sketch.jpg" http://localhost:5000/generate > out.jpg


@app.route("/generate", methods=["POST"])
def generate() -> Any:
    try:
        img = request.files.get("sketch", "")
    except Exception as err:
        return f"problem with image file: {err}", 400

    img_io = BytesIO()

    # open the image from file -> generate image
    img = to_tensor(Image.open(img))
    out = gen(img)
    out = torch.squeeze(out)

    # convert to image and write to buffer
    out_img = to_img(out)
    out_img.save(img_io, "JPEG", quality=100)
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")
