from transformer.lr_schedulers.lr_scheduler import LRScheduler


class NoamLRScheduler(LRScheduler):
    d_model: int
    wamrup_steps: int

    def __init__(self, factor: float, d_model: int, wamrup_steps: int):
        self.factor = factor
        self.d_model = d_model
        self.wamrup_steps = wamrup_steps

    def get_factor(self, step: int):
        step += 1  # Avoid zero
        return (
            self.factor
            * (self.d_model ** -0.5)
            * min(step ** (-0.5), step * self.wamrup_steps ** (-1.5))
        )
