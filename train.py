import numpy as np

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
):
    epoch_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        grad = criterion.backward()

        model.backward(grad)

        optimizer.step()

        epoch_loss += loss

    return epoch_loss / len(train_loader)


def validate_one_epoch(
    model,
    validation_loader,
    criterion,
):
    epoch_loss = 0.0

    for images, labels in validation_loader:
        outputs = model(images)

        loss = criterion(outputs, labels)

        epoch_loss += loss

    return epoch_loss / len(validation_loader)


def train(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    epochs,
):
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        validation_loss = validate_one_epoch(
            model=model,
            validation_loader=validation_loader,
            criterion=criterion,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {validation_loss:.4f}"
        )