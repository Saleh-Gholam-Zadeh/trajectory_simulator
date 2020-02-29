import logging
import copy
import torch
from crowd_sim.envs.utils.info import *

import time
from time import gmtime, strftime
jj=0
person_num=0
class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase,file_name, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        global jj,person_num

        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        #print('run_k_episode:-------------------------------------')
        #file_name = strftime("%Y-%m-%d %H:%M", gmtime())
        f = open(file_name, "a+")
        print('k = ',k,' episode is being simulated')
        time.sleep(1)
        for i in range(k):
            #if (i%20 ==1):
                #print('i =',i)
                #time.sleep(3)
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []

            #print('i/k = ',i,'/',k)

            while not done:  # yek trajectory ra ta tahesh mire
                #print('qqqqqqqqqqq')



                #f.write(str(jj*10)+"\t")

                #f.write("0" + "\t" + str(self.robot.px) + "\t" + str(self.robot.py) + "\n") --> robot invisible

                for i in range(len(ob)):
                    #print('human{} coordinate{} {}'.format(i, ob[i].px, ob[i].py))
                    f.write(str(jj * 10) + "\t")  #jj*10 -->timestamp
                    f.write(str(person_num + i + 1) + "\t" + str(round(ob[i].px,2)) + "\t" + str(round(ob[i].py,2)) + "\n")

                action = self.robot.act(ob) # human ha actioneshoon inja tavassote policy taeen va anjam mishe va felan bahash kari nadarim
                action._replace(vx=0)
                action._replace(vy=0)

                # human time tooye moteghayyere self.env hastesh

                #print(action)


                ob, reward, done, info = self.env.step(action)

                states.append(self.robot.policy.last_state)

                actions.append(action)
                rewards.append(reward)
                jj+=1

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            person_num += 100 # after each episode we create new numbers for human

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                #print('temp:reach_goal')
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                #print('temp:Collide')
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
                #print('temp:timeout')
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    #print('update memory called!  ---explorer.py line 81--')
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        # logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
        #              format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
        #                     average(cumulative_rewards)))

        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):  #  here you have access to states before transformation
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                #print('imitation learning!')
                #print('self.target_policy',self.target_policy)
                state = self.target_policy.transform(state)   #  you can remove transformation from here if it is necessary

                #print('------------------------------------debug explorer.py---line 114----------------------------')
                #print('transformation done!')
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                #print('Reinforcement')
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
