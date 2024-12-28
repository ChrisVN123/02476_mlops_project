import matplotlib.pyplot as plt
import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    # Construct the path to corruptedmnist dynamically
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/"))

    torch.save(model.state_dict(), rf"{model_path}/model.pth")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(statistics["train_loss"], label="Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid()
    # Construct the path to corruptedmnist dynamically
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/"))

    plt.savefig(rf"{img_path}/training_curve.png")
    plt.show()


if __name__ == "__main__":
    typer.run(train)
