
import gym
import os
from gym import spaces
import numpy as np
from PIL import Image, ImageDraw

class GridWorldEnv_v0(gym.Env):
    def __init__(self, grid_size=16, seed=None):
        if (seed is None):
            seed = 0
        np.random.seed(seed)
        print(f'seed = {seed}')
        self.grid_size = grid_size
        self.max_episode_length = grid_size**2

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,self.grid_size, self.grid_size), dtype=np.float32)
        self.penalty_spots = []
        point = np.array([grid_size//2,grid_size//2])
        for x in range(grid_size):
            for y in range(grid_size):
                if (np.linalg.norm(np.array([x,y]) - point)<=grid_size/4):
                    self.penalty_spots.append(np.array([x,y]))

        self.agent_pos = self._generate_random_pos(ignore_agent=True, ignore_goal=True)
        self.goal_pos = self._generate_random_pos(ignore_agent=False, ignore_goal=True)

        self.reset()

    def reset(self):
        self.num_steps = 0
        self.agent_pos = self._generate_random_pos(ignore_agent=True, ignore_goal=True)
        self.goal_pos = self._generate_random_pos(ignore_agent=False, ignore_goal=True)
        while np.all(self.agent_pos == self.goal_pos):
            self.goal_pos = self._generate_random_pos(ignore_agent=False, ignore_goal=True)
        return self._create_observation(),{}

    def generate_bfs(self):
        high_value = 10000000
        bfs_arr = np.full((self.grid_size,self.grid_size),high_value)
        bfs_arr[self.goal_pos[0],self.goal_pos[1]] = 0
        queue = [(self.goal_pos[0],self.goal_pos[1])]
        while(len(queue)):
            (x,y) = queue.pop(0)
            for action in range(4):
                newX = max(min(x + self.MOVEMENTS[action][0],self.grid_size-2),1)
                newY = max(min(y + self.MOVEMENTS[action][1],self.grid_size-2),1)
                if (newX == x and newY == y) or (self._is_penalty(np.array([newX,newY]))):
                    continue
                if (bfs_arr[newX,newY]!=high_value):
                    continue
                bfs_arr[newX,newY] = bfs_arr[x,y]+1
                queue.append((newX,newY))
        self.bfs_arr = bfs_arr
        
    def step(self, action):
        self.num_steps += 1
        new_pos = self.agent_pos + self.MOVEMENTS[action]
        old_pos = self.agent_pos
        reward = 0
        done = self.num_steps>=self.max_episode_length
        
        if self._is_valid_pos(new_pos):
            self.agent_pos = new_pos
        if np.all(self.agent_pos == self.goal_pos):
            reward = 1.0
            done = True
        else:
            old_dis = np.linalg.norm(old_pos/self.grid_size - self.goal_pos/self.grid_size)
            new_dis = np.linalg.norm(self.agent_pos/self.grid_size - self.goal_pos/self.grid_size)
            reward += (new_dis-old_dis)
            
        if self._is_penalty(self.agent_pos):
            done = True

        return self._create_observation(), reward, done,False, {}

    def expert_sample(self,random_eps = 0.0):
        highest_action = None
        for action in range(4):
            new_pos = self.agent_pos + self.MOVEMENTS[action]
            if not self._is_valid_pos(new_pos):
                continue
            if (highest_action is None):
                highest_action = action
                continue
            highest_pos = self.agent_pos + self.MOVEMENTS[highest_action]
            if (self.bfs_arr[new_pos[0],new_pos[1]]<self.bfs_arr[highest_pos[0],highest_pos[1]]):
                highest_action = action
        if (np.random.rand()<=random_eps):
            return np.random.randint(0,4)
        else:
            return highest_action
            
    def render(self, mode="human", save_path=None):
        cell_size = max(5, 500 // self.grid_size)
        img_size = cell_size * self.grid_size
        img = Image.new("RGB", (img_size, img_size), "white")
        draw = ImageDraw.Draw(img)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = np.array([i, j])
                x, y = j * cell_size, i * cell_size

                if self._is_border(pos):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill="black")
                elif np.all(pos == self.agent_pos):
                    draw.ellipse([x, y, x + cell_size, y + cell_size], fill="blue")
                elif np.all(pos == self.goal_pos):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill="green")
                elif self._is_penalty(pos):
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill="red")

        if mode == "human":
            img.show()
        elif mode == "rgb_array":
            return np.array(img)
        elif mode == "PIL":
            return img
        elif mode == "save":
            if save_path is not None:
                img.save(save_path)
            else:
                raise ValueError("save_path must be specified when mode is 'save'")

    def _create_observation(self):
        grid = np.zeros((3,self.grid_size, self.grid_size), dtype=np.float32)
        grid[0][tuple(self.agent_pos)] = 1
        grid[1][tuple(self.goal_pos)] = 1
        for spot in self.penalty_spots:
            grid[2][tuple(spot)] = 1

        return grid

    def _is_valid_pos(self, pos):
        return not self._is_border(pos)

    def _is_border(self, pos):
        return np.any(pos == 0) or np.any(pos == self.grid_size - 1)

    def _is_penalty(self, pos):
        return any(np.all(pos == spot) for spot in self.penalty_spots)

    def _generate_random_pos(self, ignore_agent=True, ignore_goal=True):
        while True:
            pos = np.random.randint(1, self.grid_size - 1, 2)
            if ((ignore_agent or not np.all(pos == self.agent_pos)) and
                (ignore_goal or not np.all(pos == self.goal_pos)) and
                not self._is_penalty(pos)):
                return pos

    def _closer_to_goal(self, old_pos):
        old_distance = np.linalg.norm(old_pos - self.goal_pos)
        new_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        return new_distance < old_distance

    MOVEMENTS = {
        0: np.array([-1, 0]),  # up
        1: np.array([0, 1]),   # right
        2: np.array([1, 0]),   # down
        3: np.array([0, -1]),  # left
    }

gym.envs.registration.register(
    id='GridWorld-v0',
    entry_point='env:GridWorldEnv_v0',
    kwargs={}
)

if __name__=='__main__':
    env = gym.make('GridWorld-v0', grid_size=16,seed=43)
    state_shape = env.observation_space.shape
    action_shape = (1,)

    from tqdm import trange
    from buffer import Buffer
    
    buffer_size = 10000
    expert_buffer = Buffer(buffer_size=buffer_size,state_shape=state_shape,
                           action_shape=action_shape,device='mps')
    returns = []
    render = False
    num_traj = 10000
    if (render):
        num_traj = 5
        os.makedirs('expert_gifs',exist_ok=True)
    for ep in trange(num_traj):
        state,_ = env.reset()
        env.generate_bfs()
        frames = []
        while(True):
            if (render):
                img = env.render(mode="PIL",)
                frames.append(img)
            action = env.expert_sample(random_eps = 0.0)
            next_state, reward, done,_,_ = env.step(action)

            if (not render):
                expert_buffer.append(state=state,action=np.array([action]),next_state=next_state,
                                    done=np.array([done]))
                if (expert_buffer._n == expert_buffer.total_size):
                    expert_buffer.save(f'dataset/dataset_{buffer_size}.pth')
                    print(np.mean(returns))
                    print('done')
                    exit()
            if (done):
                break
            state = next_state
        returns.append(reward)
        
        if (render):
            img_save_path = f"expert_gifs/{ep}.gif"
            frames[0].save(img_save_path, format="GIF", append_images=frames,
                    save_all=True, duration=100, loop=0)
            print(f'render at {img_save_path}')
    env.close()