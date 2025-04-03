from agents.base_agent import BaseAgent

class BeliefAgent(BaseAgent):
    def __init__(self, config):
        super(BeliefAgent, self).__init__(config)
        assert not self.decode_reward
        assert self.decode_task
        assert not self.contrastive_task_loss
        assert self.use_kl_loss
