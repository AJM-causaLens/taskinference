from agents.base_agent import BaseAgent

class VaribadAgent(BaseAgent):
    def __init__(self, config):
        super(VaribadAgent, self).__init__(config)
        assert self.decode_reward
        assert not self.decode_task
        assert not self.contrastive_task_loss
        assert self.use_kl_loss

