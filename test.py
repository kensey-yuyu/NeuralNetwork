import argparse

import torch

from config import Config
from nn import NeuralNetwork
from utils import evaluate, load_dataset


def main() -> None:
    """
    Test for trained model. 
    """

    # Get path to trained model.
    parser = argparse.ArgumentParser()
    parser.add_argument("trained_model_path", type=str,
                        help="Path of trained model.")
    args = parser.parse_args()

    # Model.
    trained_model = NeuralNetwork()
    trained_model_data = torch.load(args.trained_model_path)
    trained_model.load_state_dict(trained_model_data["model_state_dict"])

    # Init config.
    cfg = Config(trained_model)
    trained_model.to(cfg.device)

    # Test.
    # Load dataset.
    test_loader = load_dataset(
        root=cfg.root, train=False, transform=cfg.transform, batch_size=cfg.batch_size)
    # Evaluate.
    test_accuracy, test_loss = evaluate(trained_model, test_loader, cfg)
    print(f"Test accuracy: {test_accuracy:.3f}, Test loss: {test_loss:.3f}")
    return


if __name__ == "__main__":
    main()
