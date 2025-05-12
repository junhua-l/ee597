
### Dynamic Spectrum Access Using DRL

EE597 Final Project
Dynamic Spectrum Access Using Deep Reinforcement Learning Networks Proposal
Zhaokun Jian, Junhua Liu

Dynamic spectrum access (DSA) aims to alleviate the spectrum scarcity by allowing unlicensed devices to exploit the licensed channels. Realising DSA in random multichannel access networks is challenging because the users receive only binary acknowledgements, have no knowledge of other users’ actions, and must operate without control signalling. Traditional protocols (such as Slotted Aloha) are inefficient due to collisions and idle time slots, especially in distributed and large-scale environments. In this paper, we aim to find a multi-user strategy to maximize the network utility without online coordination. We treat multichannel DSA as a partially observable stochastic game and propose a multi-user deep reinforcement-learning (DRL) solution that is both scalable and fully distributed. Specifically, each user executes an identical Deep Q-Network (DQN) augmented with a Long Short-Term Memory (LSTM) layer to learn channel access policies, enabling it to infer latent network dynamics from short acknowledgement histories.  All neural parameters are trained offline at a central server with double Q-learning. The simulations highlight the effectiveness of proposed DRL framework in achieving adaptive and distributed spectrum management without explicit adaptation and deploying learning-based spectrum sharing.

---

## 1. Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access.git
   cd Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\Scripts\activate      # on Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install matplotlib tensorflow numpy jupyter
   ```

------

## 2. Usage

To train the Deep Q-Network (DQN) on your default environment, simply run:

```bash
python train.py
```

This will:

- Initialize the multi-user network environment
- Build and train the DRQN model
- Periodically output training loss and cumulative reward plots into `figures/`

------

## 3. File Overview

- **`drqn.py`**
   Implements the Deep Recurrent Q-Network (DRQN) model architecture:
  - Input layer encoding last action, channel capacities, and ACK
  - LSTM layer for partial observability
  - Dueling-DQN branches (value & advantage streams)
  - Double Q-learning target updates 
- **`multi_user_network.py`**
   Defines the simulated multi-user, multi-channel environment:
  - `EnvNetwork` class handles random access slots, collision logic, and ACK feedback
  - Supports configurable number of users NN and channels KK
- **`train.py`**
   Coordinates training:
  - Parses hyperparameters (episodes, time slots, learning rate, etc.)
  - Runs the offline centralized DRQN training loop (no experience replay)
  - Saves model checkpoints and training plots under `figures/`

------

## 4. References 
(Organized by published order)

1. V. Mnih *et al.*, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, no. 7540, pp. 529–533, Feb. 2015.  
2. M. Hausknecht and P. Stone, “Deep recurrent Q-learning for partially observable MDPs,” *arXiv preprint arXiv:1507.06527*, Jul. 2015.
3. Z. Qin, W. Saad, and T. Başar, “A Survey on Dynamic Spectrum Access in Cognitive Radio Networks: Insights, Approaches and Open Problems,” *IEEE Communications Surveys & Tutorials*, vol. 18, no. 1, pp. 350–379, Firstquarter 2016.
4. X. Wang, R. Chen, T. Taleb, A. Ksentini, and V. Leung, “Dynamic Service Placement for Mobile Micro-Clouds with Predicted Future Costs,” *IEEE Transactions on Parallel and Distributed Systems*, vol. 28, no. 4, pp. 1002–1016, Apr. 2017.  
5. X. Wang, R. Chen, T. Taleb, A. Ksentini, and V. Leung, “Dynamic Service Placement for Mobile Micro-Clouds with Predicted Future Costs,” *IEEE Transactions on Parallel and Distributed Systems*, vol. 28, no. 4, pp. 1002–1016, Apr. 2017.      
6. O. Naparstek and K. Cohen, “Deep Multi-User Reinforcement Learning for Dynamic Spectrum Access in Multichannel Wireless Networks,” *IEEE Transactions on Wireless Communications*, vol. 7, no. 1, pp. 604–617, Jan. 2018.   
7. Y. Li, R. Wang, X. Zhang, and F. R. Yu, “Deep Reinforcement Learning for Dynamic Spectrum Access: Model, Algorithm, and Evaluation,” in *Proc. IEEE ICC*, Shanghai, China, May 2019, pp. 1–6.  
8. T. Nguyen and T. Sugimoto, “Cooperative Multi-Agent Reinforcement Learning for Spectrum Sharing in 5G,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 5, no. 3, pp. 633–646, Sep. 2019.  
9. N. Kato *et al.*, “An Overview of Deep Learning in Satellite and Terrestrial Communications,” *IEEE Communications Surveys & Tutorials*, vol. 21, no. 4, pp. 2943–2974, Fourthquarter 2019.  

