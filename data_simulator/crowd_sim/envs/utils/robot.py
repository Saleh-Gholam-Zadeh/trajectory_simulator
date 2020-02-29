from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)


        # print('----debug robot.py line 15 ')
        # print('robot cordinate = [',self.px,',',self.py,']')





        action = self.policy.predict(state)
        #IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!    CHNGE IT IF YOU DONT WANT TO CREATE SIMULATION DATA
        action = ActionXY(0,0)
        return action
