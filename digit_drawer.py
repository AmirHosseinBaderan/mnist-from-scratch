import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw


class DigitDrawer:

    def __init__(self, model):
        self.model = model

        self.size = 280
        self.brush_size = 18

        self.root = tk.Tk()
        self.root.title("MNIST Predictor")

        self.canvas = tk.Canvas(
            self.root,
            width=self.size,
            height=self.size,
            bg="black"
        )
        self.canvas.pack()

        self.result_label = tk.Label(
            self.root,
            text="Draw a digit",
            font=("Arial", 20)
        )
        self.result_label.pack()

        self.predict_button = tk.Button(
            self.root,
            text="Predict",
            command=self.predict
        )
        self.predict_button.pack()

        self.clear_button = tk.Button(
            self.root,
            text="Clear",
            command=self.clear
        )
        self.clear_button.pack()

        self.image = Image.new(
            "L",
            (self.size, self.size),
            color=0
        )

        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x = event.x
        y = event.y

        r = self.brush_size

        self.canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            fill="white",
            outline="white"
        )

        self.draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=255
        )

    def clear(self):
        self.canvas.delete("all")

        self.image = Image.new(
            "L",
            (self.size, self.size),
            color=0
        )

        self.draw = ImageDraw.Draw(self.image)

        self.result_label.config(text="Draw a digit")

    def predict(self):
        img = self.image.resize((28, 28))

        x = np.array(img, dtype=np.float32)

        x /= 255.0

        x = x.reshape(1, 784)

        logits = self.model(x)

        prediction = np.argmax(logits, axis=1)[0]

        self.result_label.config(
            text=f"Prediction: {prediction}"
        )

    def run(self):
        self.root.mainloop()