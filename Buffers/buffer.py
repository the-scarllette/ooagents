import numpy as np

class Buffer:
    def __init__(
        self,
        state_dim,
        act_dim,
        max_size,
        min_data=256
    ):
        self.mem_size = int(max_size)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.counter = 0

        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, act_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
        self._populated = False
        self.min_data = min_data
        
    def clear_buffer(
        self
    ) -> None:
        
        self.__init__(self.state_dim, self.act_dim, self.mem_size, self.min_data)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        index = self.counter % self.mem_size

        l = state.shape[0]
        space = self.mem_size - index
        remainder = l - space
        if space < l:
            # * Handle overflow
            self.state_memory[index:] = state.reshape(-1, self.state_dim)[:space]
            self.action_memory[index:] = action.reshape(-1, self.act_dim)[:space]
            self.next_state_memory[index:] = next_state.reshape(-1, self.state_dim)[:space]
            self.reward_memory[index:] = reward[:space]
            self.terminal_memory[index:] = terminal[:space]
            
            self.state_memory[:remainder] = state.reshape(-1, self.state_dim)[space:]
            self.action_memory[:remainder] = action.reshape(-1, self.act_dim)[space:]
            self.next_state_memory[:remainder] = next_state.reshape(-1, self.state_dim)[space:]
            self.reward_memory[:remainder] = reward[space:]
            self.terminal_memory[:remainder] = terminal[space:]
            
            self.counter += l
            # self.counter %= self.mem_size
            
        else:
            self.state_memory[index:index+l, ...] = state.reshape(-1, self.state_dim)
            self.action_memory[index:index+l, ...] = action.reshape(-1, self.act_dim)
            self.next_state_memory[index:index+l, ...] = next_state.reshape(-1, self.state_dim)
            self.reward_memory[index:index+l] = reward
            self.terminal_memory[index:index+l] = terminal
            
            self.counter += l
            
        if not self._populated:
            if self.counter >= self.min_data:
                self._populated = True

    def sample_batch(
        self,
        batch_size
    ) -> None:
        
        sample_size = min(self.mem_size, self.counter)
        batch = np.random.choice(sample_size, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, next_states, terminals

    def get_experiences(
        self,
        indices
        ):
        
        return self.state_memory[indices], self.action_memory[indices], self.reward_memory[indices], self.next_state_memory[indices], self.terminal_memory[indices]
    
    @property
    def populated(self):
        return self._populated
    
class OptionBuffer(Buffer):
    def __init__(self, state_dim, act_dim, max_size, min_data=256):
        super().__init__(state_dim, act_dim, max_size, min_data=256)
        self.step_memory = np.zeros(self.mem_size, dtype=np.int32)
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, terminal: bool, steps: np.ndarray):
        index = self.counter % self.mem_size
        l = state.shape[0]
        space = self.mem_size - index
        remainder = l - space
        if space < l:
            # * Handle overflow
            self.state_memory[index:] = state.reshape(-1, self.state_dim)[:space]
            self.action_memory[index:] = action.reshape(-1, self.act_dim)[:space]
            self.next_state_memory[index:] = next_state.reshape(-1, self.state_dim)[:space]
            self.reward_memory[index:] = reward[:space]
            self.terminal_memory[index:] = terminal[:space]
            
            self.state_memory[:remainder] = state.reshape(-1, self.state_dim)[space:]
            self.action_memory[:remainder] = action.reshape(-1, self.act_dim)[space:]
            self.next_state_memory[:remainder] = next_state.reshape(-1, self.state_dim)[space:]
            self.reward_memory[:remainder] = reward[space:]
            self.terminal_memory[:remainder] = terminal[space:]
            self.step_memory[:remainder] = steps[space:]
            
            self.counter += l
            # self.counter %= self.mem_size
            
        else:
            self.state_memory[index:index+l, ...] = state.reshape(-1, self.state_dim)
            self.action_memory[index:index+l, ...] = action.reshape(-1, self.act_dim)
            self.next_state_memory[index:index+l, ...] = next_state.reshape(-1, self.state_dim)
            self.reward_memory[index:index+l] = reward
            self.terminal_memory[index:index+l] = terminal
            self.step_memory[index:index+l] = steps
            
            self.counter += l
            
        if not self._populated:
            if self.counter >= self.min_data:
                self._populated = True
                
    def sample_batch(self, batch_size):
        sample_size = min(self.mem_size, self.counter)
        batch = np.random.choice(sample_size, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]
        steps = self.step_memory[batch]
        return states, actions, rewards, next_states, terminals, steps
        
    def get_experiences(self, indices):
        return self.state_memory[indices], self.action_memory[indices], self.reward_memory[indices], self.next_state_memory[indices], self.terminal_memory[indices], self.step_memory[indices]
    