def create_constant_learning_rate_schedule(
    base_learning_rate,
    steps_per_epoch,
    warmup_length: float = 0.0,
): ...
def create_stepped_learning_rate_schedule(
    base_learning_rate,
    steps_per_epoch,
    lr_sched_steps,
    warmup_length: float = 0.0,
): ...
def create_cosine_learning_rate_schedule(
    base_learning_rate,
    steps_per_epoch,
    halfcos_epochs,
    warmup_length: float = 0.0,
): ...
