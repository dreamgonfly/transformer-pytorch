from torch.optim import Optimizer, Adam


class NoamOptimizer(Optimizer):

    def __init__(self, params, d_model, warmup_steps=4000, betas=(0.9, 0.98), eps=-1e9):
        self.optimizer = Adam(params, betas=betas, eps=eps)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        super(NoamOptimizer, self).__init__()

    def step(self, clousre=None):
        self.step_num += 1
        self.lr = self.lrate()
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr
        self.optimizer.step()

    def lrate(self):
        return self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
