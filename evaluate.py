import numpy as np


def evaluate(
    model,
    test_loader,
    criterion,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        predictions = np.argmax(outputs, axis=1)

        correct += np.sum(predictions == labels)

        total += len(labels)

        total_loss += loss

    return (
        total_loss / len(test_loader),
        correct / total,
    )