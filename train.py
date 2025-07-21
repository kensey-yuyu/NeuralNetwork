from tqdm import tqdm

from config import Config
from nn import NeuralNetwork
from utils import Logger, evaluate, load_dataset, train


def main() -> None:
    """
    Neural Network training.
    """

    # Model.
    model = NeuralNetwork()

    # Init config and logger.
    cfg = Config(model)
    model.to(cfg.device)
    logger = Logger(cfg, model)

    # Training.
    # Load dataset.
    train_loader = load_dataset(
        root=cfg.root, train=True, transform=cfg.transform, batch_size=cfg.batch_size)
    test_loader = load_dataset(
        root=cfg.root, train=False, transform=cfg.transform, batch_size=cfg.batch_size)
    # Training loop.
    for epoch in tqdm(range(cfg.epoch), desc="Training"):
        train_accuracy, train_loss = train(model, train_loader, cfg)
        test_accuracy, test_loss = evaluate(model, test_loader, cfg)
        logger.record(epoch, train_accuracy, train_loss,
                      test_accuracy, test_loss)
        tqdm.write(
            f"Epoch: {epoch:{len(str(cfg.epoch))}}, Train accuracy: {train_accuracy:.3f}, Train loss: {train_loss:.3f}, Test accuracy: {test_accuracy:.3f}, Test loss: {test_loss:.3f}")
    return


if __name__ == "__main__":
    main()
