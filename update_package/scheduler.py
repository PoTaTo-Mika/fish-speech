import math


def get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int | float,
    num_training_steps: int,
    num_cycles: float = 0.5,
    final_lr_ratio: float = 0.0,
):
    if 0 < num_warmup_steps < 1:  # float mode
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_training_steps:
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        value = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return final_lr_ratio + value * (1.0 - final_lr_ratio)
    else:
        return final_lr_ratio


def get_constant_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int | float,
    num_training_steps: int | None = None,
    num_zero_warmup_steps: int | float = 0,
):
    if 0 < num_warmup_steps < 1:  # float mode
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    if 0 < num_zero_warmup_steps < 1:  # float mode
        num_zero_warmup_steps = int(num_zero_warmup_steps * num_warmup_steps)

    assert (
        num_zero_warmup_steps <= num_warmup_steps
    ), "num_zero_warmup_steps must be less than or equal to num_warmup_steps"

    if current_step < num_zero_warmup_steps:
        return 0.0
    elif current_step < num_warmup_steps:
        return float(current_step - num_zero_warmup_steps) / float(
            num_warmup_steps - num_zero_warmup_steps
        )

    return 1.0


def get_wsd_schedule_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int | float,
    num_decay_steps: int | float,
    num_training_steps: int,
    num_cycles: float = 0.5,
    final_lr_ratio: float = 0.0,
):
    if isinstance(num_warmup_steps, float):
        num_warmup_steps = int(num_warmup_steps * num_training_steps)
    if isinstance(num_decay_steps, float):
        num_decay_steps = int(num_decay_steps * num_training_steps)

    num_constant_steps = num_training_steps - num_warmup_steps - num_decay_steps

    constant_end = num_warmup_steps + num_constant_steps
    decay_end = constant_end + num_decay_steps

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < constant_end:
        return 1.0
    elif current_step < decay_end:
        progress = float(current_step - num_warmup_steps - num_constant_steps) / float(
            max(1, num_decay_steps)
        )
        value = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return (1.0 - final_lr_ratio) * value + final_lr_ratio
    else:
        return final_lr_ratio


def get_stable_lr_lambda(current_step: int):
    return 1.0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    plt.switch_backend("Agg")

    # Set up parameters
    num_training_steps = 100000
    num_warmup_steps = 1000
    final_lr_ratio = 0.1

    # Generate steps
    steps = np.arange(num_training_steps)

    # Calculate learning rates for each scheduler
    cosine_lrs = [
        get_cosine_schedule_with_warmup_lr_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            final_lr_ratio=final_lr_ratio,
        )
        for step in steps
    ]

    constant_lrs = [
        get_constant_schedule_with_warmup_lr_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        for step in steps
    ]

    wsd_lrs = [
        get_wsd_schedule_lr_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=0.1,
            num_training_steps=num_training_steps,
            final_lr_ratio=final_lr_ratio,
        )
        for step in steps
    ]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cosine_lrs, label="Cosine with Warmup")
    plt.plot(steps, constant_lrs, label="Constant with Warmup")
    plt.plot(steps, wsd_lrs, label="WSD")

    plt.xlabel("Steps")
    plt.ylabel("Learning Rate Multiplier")
    plt.title("Learning Rate Schedules")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_schedules.png")
    plt.close()
