class Config:
    """Config

    Configurations for learning.

    Attributes:
        device: Device for running (CPU or GPU).
        seed: Seed value.
        epoch: Epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        input_size: Input size of model.
        transform: Transforms for dataset (Data augmentation etc).
        criterion: Loss function.
        optimizer: Optimizer function.
        scheduler: Scheduler for update learning rate. 
        weight_decay: Weight decay.

    """

    def __init__(self) -> None:
        """__init__

        Initialize Config.

        """

        self.device = None
        self.seed = None
        self.epoch = None
        self.batch_size = None
        self.learning_rate = None
        self.input_size = None
        self.transform = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.weight_decay = None
        return
