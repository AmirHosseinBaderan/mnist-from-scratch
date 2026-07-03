import numpy as np


def predict(model, data_loader, count=10):
    model.eval()

    shown = 0

    for images, labels in data_loader:

        outputs = model(images)

        predictions = np.argmax(outputs, axis=1)

        for image, prediction, label in zip(images, predictions, labels):

            print(f"Prediction: {prediction} | Label: {label}")

            shown += 1

            if shown >= count:
                return