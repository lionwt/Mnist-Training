import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tkinter as tk
import torch
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms
from util.device import get_device
from Deeplearning.model import NeuralNetwork


# -------------------------
# Load model
# -------------------------
device = get_device()

model = NeuralNetwork().to(device)
model.load_state_dict(
  torch.load("model.pth", map_location="cpu", weights_only=False)
)
model.to(device)


# -------------------------
# Drawing canvas
# -------------------------
CANVAS_SIZE = 280

root = tk.Tk()
root.title("MNIST Digit Drawer")

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black"
)
canvas.pack()

# PIL image corresponding to the canvas
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)


# -------------------------
# Mouse drawing
# -------------------------
def draw_digit(event):
    x = event.x
    y = event.y

    radius = 10

    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ),
        fill=255
    )


canvas.bind("<B1-Motion>", draw_digit)


PREVIEW_SCALE = 8  # 28x28 shown as 224x224 so each model pixel is visible


def model_input_image():
    return image.resize((28, 28), Image.Resampling.NEAREST)


def show_model_input(img_28):
    preview = img_28.resize(
        (28 * PREVIEW_SCALE, 28 * PREVIEW_SCALE),
        Image.Resampling.NEAREST,
    )
    photo = ImageTk.PhotoImage(preview)
    preview_label.configure(image=photo)
    preview_label.image = photo  # keep a reference


# -------------------------
# Predict
# -------------------------
def predict():
    img = model_input_image()
    show_model_input(img)

    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()

    result_label.config(text=str(prediction))


# -------------------------
# Clear canvas
# -------------------------
def clear():
    canvas.delete("all")

    draw.rectangle(
        (0, 0, CANVAS_SIZE, CANVAS_SIZE),
        fill=0
    )

    result_label.config(text="-")
    show_model_input(model_input_image())


# -------------------------
# Buttons
# -------------------------
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict
)
predict_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear
)
clear_button.pack(side=tk.LEFT, padx=5)


result_frame = tk.Frame(root)
result_frame.pack(pady=10)

preview_label = tk.Label(result_frame, bg="black")
preview_label.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(
    result_frame,
    text="-",
    font=("Arial", 72),
)
result_label.pack(side=tk.LEFT, padx=10)

show_model_input(model_input_image())


root.mainloop()