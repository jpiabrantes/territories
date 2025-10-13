from typing import Literal, Optional
import gymnasium
import numpy as np

import pufferlib
import binding


def set_buffers(env, buf=None):
    if buf is None:
        obs_space = env.single_observation_space
        env.observations = np.zeros((env.num_agents, *obs_space.shape), dtype=obs_space.dtype)
        env.rewards = np.zeros(env.num_agents, dtype=np.float32)
        env.terminals = np.zeros(env.num_agents, dtype=bool)
        env.truncations = np.zeros(env.num_agents, dtype=bool)
        env.alive_mask = np.zeros(env.num_agents, dtype=bool)
        # TODO: this could be replaced with just the upper triangle (but it seems like the added complexity is not worth it)
        env.kinship_matrix = np.zeros((env.num_envs*env.agents_per_env*env.agents_per_env), dtype=np.uint8)

        # TODO: Major kerfuffle on inferring action space dtype. This needs some asserts?
        atn_space = pufferlib.spaces.joint_space(env.single_action_space, env.num_agents)
        if isinstance(env.single_action_space, pufferlib.spaces.Box):
            env.actions = np.zeros(atn_space.shape, dtype=atn_space.dtype)
        else:
            env.actions = np.zeros(atn_space.shape, dtype=np.int32)
    else:
        env.observations = buf['observations']
        env.rewards = buf['rewards']
        env.terminals = buf['terminals']
        env.truncations = buf['truncations']
        env.alive_mask = buf['alive_mask']
        env.actions = buf['actions']
        env.kinship_matrix = buf['kinship_matrix']



class Territories(pufferlib.PufferEnv):
    def __init__(self, num_envs: int=1, agents_per_env: int=512, n_genes: int=3, n_roles: int=2,
                 width: int=96, height: int=96, min_ep_length: int=512, max_ep_length: int=576, extinction_reward: float=-2.0,
                 render_mode: Literal["normal", "replay"]="normal", buf=None, seed: int=0, log_interval: int=100):
        self.log_interval = log_interval
        self.num_envs = num_envs
        self.n_genes = n_genes
        self.n_roles = n_roles
        self.num_agents = agents_per_env * num_envs
        self.agents_per_env = agents_per_env
        self.n_roles = n_roles
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255, 
                                                             shape=(9*9*(10+n_genes)+5+n_genes+5,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(11)
        self.infos: list[dict] = []
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.num_agents)
        self.agent_ids = np.arange(self.num_agents)
        set_buffers(self, buf)
        
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(self.observations[i*agents_per_env:(i+1)*agents_per_env],
                                     self.actions[i*agents_per_env:(i+1)*agents_per_env], 
                                     self.rewards[i*agents_per_env:(i+1)*agents_per_env], 
                                     self.terminals[i*agents_per_env:(i+1)*agents_per_env],
                                     self.truncations[i*agents_per_env:(i+1)*agents_per_env],
                                     seed,
                                     self.alive_mask[i*agents_per_env:(i+1)*agents_per_env],
                                     self.kinship_matrix[i*agents_per_env*agents_per_env:(i+1)*agents_per_env*agents_per_env],
                                     render_mode=0 if render_mode == "normal" else 1,
                                     extinction_reward=extinction_reward,
                                     n_roles=n_roles, n_genes=n_genes, width=width, height=height,
                                     max_agents=agents_per_env, min_ep_length=min_ep_length, max_ep_length=max_ep_length)
            c_envs.append(c_env)
        self.c_envs = binding.vectorize(*c_envs)
        self.env = None
        if num_envs == 1:
            self.env = c_envs[0]

    def reset(self, seed: int=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, self.alive_mask

    def step(self, actions: Optional[np.ndarray] = None, acting_mask: Optional[np.ndarray] = None, actions_are_set: bool = False):
        self.tick += 1
        if not actions_are_set:
            if acting_mask is None:
                self.actions[self.alive_mask] = actions
            else:
                self.actions[acting_mask] = actions
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
                
        return (self.observations, self.rewards,
            self.terminals, self.truncations, self.alive_mask, self.kinship_matrix, info)
        
    def recv(self):
        return (self.observations, self.rewards,
            self.terminals, self.truncations, self.alive_mask, self.kinship_matrix, self.info, 0)
        
    def send(self, actions, acting_mask = None):
        o, r, d, t, alive_mask, kinship_matrix, self.infos = self.step(actions, acting_mask)
        assert isinstance(self.infos, list), 'PufferEnvs must return info as a list of dicts'
        return (o, r, d, t, alive_mask, kinship_matrix, self.infos)
        
    def render(self):
        return binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)
        
    def get(self):
        assert self.env is not None, "self.env is None!"
        return binding.env_get(self.env)
    
    def put(self, **kwargs):
        assert self.env is not None, "self.env is None!"
        return binding.env_put(self.env, **kwargs)



def test_performance(cls, timeout=10, atn_cache=1024):
    env = cls(num_envs=1)
    o, alive_mask = env.reset()
    tick = 0
    steps = 0 

    actions = np.random.randint(0, 2, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache][alive_mask]
        r = env.step(atn)
        alive_mask = r[4]
        steps += alive_mask.sum()
        tick += 1

    print(f'{env.__class__.__name__}: SPS: {steps / (time.time() - start)}')


                
if __name__ == "__main__":
    TEST_PERFORMANCE = False
    if TEST_PERFORMANCE:
        from sys import exit
        test_performance(Territories) # ~1.35M SPS (NMMO3 is ~1.95M) Number of envs does not matter much but less is better I think
        exit()
    
    env = Territories(num_envs=1, agents_per_env=128, n_genes=3)
    o, alive_mask = env.reset()

    env.render()
    while True:
        o = o[alive_mask]
        actions = np.random.randint(0, 11, (o.shape[0],))
        
        o, r, d, truncations, alive_mask, kinship_matrix, i = env.step(actions)
        
        
        for _ in range(60):
            env.render()
        
        if alive_mask.sum() == 0 or truncations.sum() > 0:
            print('Game over')
            break