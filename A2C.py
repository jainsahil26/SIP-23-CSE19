# %%
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import torch.distributions as dist
from torch.distributions import Binomial

# %%
# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 1
gamma = 0.98
max_train_steps = 60000
PRINT_INTERVAL = 1 # update_interval * 100

# %%
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 2, kernel_size = 3, stride = 1, padding = 1)
        self.fc1= nn.Linear(2*96*96, 100)
        self.fc2 = nn.Linear(100 , 5)
        self.fc_v = nn.Linear(100, 1)

    def pi(self, x, softmax_dim=1):
        single_image_input = len(x.shape) == 3  # missing 4th batch dimension
        if single_image_input:
            x = x.unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)

        x=self.conv1(x)
        x= torch.relu(x)
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.conv3(x)
        x=torch.relu(x)
        x=self.conv4(x)
        x=torch.relu(x)
        x=self.conv5(x)
        x=torch.relu(x)
        x=x.reshape(x.size(0), -1)
        x= self.fc1(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)
        return x


    def v(self, x):
        x = x.permute(0, 3, 1, 2)
        x=self.conv1(x)
        x= torch.relu(x)
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.conv3(x)
        x=torch.relu(x)
        x=self.conv4(x)
        x=torch.relu(x)
        x=self.conv5(x)
        x=torch.relu(x)
        x=x.reshape(x.size(0), -1)
        x= self.fc1(x)
        v = self.fc_v(x)
        return v

# %%
device= 'cuda:0'
model = ActorCritic().to(device)
model

# %%
def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make("CarRacing-v2",continuous=False)
    #env.seed(worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            #ob, reward, done, info = env.step(data)
            print(f"data:{data}")
            ob, reward, done, truncated, info = env.step(data)
            if done:
                ob, info = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            print("resetting")
            ob, info = env.reset()
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


# %%
class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker, args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True


# %%
def test(step_idx, model, device):
    env = gym.make('CarRacing-v2',continuous=False)
    score = 0.0
    num_test = 10

    for _ in range(num_test):
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            
            prob = model.pi(torch.from_numpy(observation).float().to(device), softmax_dim=0)

            doNothing=dist.Bernoulli(prob[0][0])
            steerLeft=dist.Bernoulli(prob[0][1])
            steerRight=dist.Bernoulli(prob[0][2])
            gas=dist.Bernoulli(prob[0][3])
            brake=dist.Bernoulli(prob[0][4])

            doNothing_action=doNothing.sample().item()
            steerLeft_action=steerLeft.sample().item()
            steerRight_action=steerRight.sample().item()
            gas_action=gas.sample().item()
            brake_action=brake.sample().item()

            action=[doNothing_action,steerLeft_action,steerRight_action,gas_action,brake_action]


            
            observation_prime, reward, terminated, truncated, info = env.step(action)
            observation = observation_prime
            score += reward

        terminated = False
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}")

    env.close()

# %%
def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

# %%
if __name__ == '__main__':
    mp.set_start_method('spawn')
    if torch.cuda.is_available():
        device= 'cuda:0'
    else:
        device = 'cpu'
        
    print('device:{}'.format(device))
    envs = ParallelEnv(n_train_processes)


    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    step_idx = 0
 
    s = envs.reset()

    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(update_interval):
          
            prob = model.pi(torch.from_numpy(s).float().to(device))

            distribution=Binomial(prob)
            """
            doNothing=dist.Bernoulli(prob[0][0])
            steerLeft=dist.Bernoulli(prob[0][1])
            steerRight=dist.Bernoulli(prob[0][2])
            gas=dist.Bernoulli(prob[0][3])
            brake=dist.Bernoulli(prob[0][4])

            doNothing_action=int(doNothing.sample().item())
            steerLeft_action=int(steerLeft.sample().item())
            steerRight_action=int(steerRight.sample().item())
            gas_action=int(gas.sample().item())
            brake_action=int(brake.sample().item())
            """
            sample=distribution.sample()
            doNothing=sample[0][0]
            steerLeft=sample[0][1]
            steerRight=sample[0][2]
            gas=sample[0][3]
            brake=sample[0][4]


            a=[doNothing,steerLeft,steerRight,gas,brake]

            s_prime, r, done, info = envs.step(a)
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r/100.0)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = torch.from_numpy(s_prime).float().to(device)
        v_final = model.v(s_final).detach().cpu().clone().numpy()
        td_target = compute_target(v_final, r_lst, mask_lst)

        td_target_vec = td_target.reshape(-1)
        # s_vec = torch.tensor(s_lst).float().reshape(-1, 4).to(device)  # 4 == Dimension of state
        s_lst = [torch.from_numpy(x).to(device) for x in s_lst]
        s_vec = torch.stack(s_lst, dim=0)
        s_vec = s_vec.view(s_vec.shape[0] * s_vec.shape[1], 96, 96, 3).to(dtype=torch.float32)
        print("SVEC shape")
        print(s_vec.shape)
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1).to(device)
        advantage = td_target_vec.to(device) - model.v(s_vec).reshape(-1)
       

        pi = model.pi(s_vec, softmax_dim=1)
        """
        doNothing=dist.Bernoulli(pi[0][0])
        steerLeft=dist.Bernoulli(pi[0][1])
        steerRight=dist.Bernoulli(pi[0][2])
        gas=dist.Bernoulli(pi[0][3])
        brake=dist.Bernoulli(pi[0][4])

        doNothing_action=doNothing.sample().item()
        steerLeft_action=steerLeft.sample().item()
        steerRight_action=steerRight.sample().item()
        gas_action=gas.sample().item()
        brake_action=brake.sample().item()


      
        doNothing_all_actions = [sublist[0] for sublist in a_lst]
        doNothing_logprobs = doNothing.log_prob(torch.tensor(doNothing_all_actions))

        steerLeft_all_actions = [sublist[1] for sublist in a_lst]
        steerLeft_logprobs = steerLeft.log_prob(torch.tensor(steerLeft_all_actions))

        steerRight_all_actions = [sublist[2] for sublist in a_lst]
        steerRight_logprobs = steerRight.log_prob(torch.tensor(steerRight_all_actions))

        gas_all_actions = [sublist[3] for sublist in a_lst]
        gas_logprobs = gas.log_prob(torch.tensor(gas_all_actions))

        brake_all_actions = [sublist[4] for sublist in a_lst]
        brake_logprobs = brake.log_prob(torch.tensor(brake_all_actions))
        """
        



        """
        # Given probabilities
        probabilities = 

        # Create a binomial distribution
        binomial_dist = Binomial(probs=probabilities)

        # Generate 5 random values
        random_values = binomial_dist.sample((1,))

        print(random_values)
        """


        pi_a = pi.gather(1, a_vec).reshape(-1)
        loss = -(torch.log(pi_a).to(device) * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.v(s_vec).reshape(-1).to(device), td_target_vec.to(device))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model, device)

    envs.close()



# %%
