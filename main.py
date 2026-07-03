from config import *
from config import IMAGE_COLS

from data.dataset import MNISTDataset
from data.loader import DataLoader
from digit_drawer import DigitDrawer
from predict import predict
from transforms.flatten import Flatten

from transforms.compose import Compose
from transforms.normalize import Normalize
from transforms.to_numpy import ToNumpy
from transforms.reshape import Reshape

from nn.linear import Linear
from nn.relu import ReLU
from nn.sequential import Sequential

from nn.losses.cross_entropy import CrossEntropyLoss

from optim.sgd import SGD

from train import train

transform = Compose([
    ToNumpy(),
    Reshape(IMAGE_ROWS, IMAGE_COLS),
    Flatten(),
    Normalize(),
])

train_dataset = MNISTDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS,
    image_transform=transform,
)

test_dataset = MNISTDataset(
    TEST_IMAGES,
    TEST_LABELS,
    image_transform=transform,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = Sequential(
    Linear(INPUT_SIZE, HIDDEN_SIZE),
    ReLU(),
    Linear(HIDDEN_SIZE, NUM_CLASSES),
)

criterion = CrossEntropyLoss()

optimizer = SGD(
    model.parameters(),
    learning_rate=LEARNING_RATE,
)

train(
    model=model,
    train_loader=train_loader,
    validation_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=EPOCHS,
)

# print("\n========== Prediction ==========\n")
#
# predict(
#     model=model,
#     data_loader=test_loader,
#     count=20,
# )

app = DigitDrawer(model)
app.run()