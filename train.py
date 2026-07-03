def train(
        model,
        train_loader,
        criterion,
        optimizer,
        epochs,
):
    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            grad = criterion.backward()
            model.backward(grad)

            optimizer.step()

            epoch_loss += loss

        epoch_loss /= len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}]"
            f"Loss : {epoch_loss:.4f}"
        )
