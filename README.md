# Landing On The Moon Project
# Authors
[Coursea] Reinforcement Learning Specialization by "University of Alberta" & "Alberta Machine Intelligence Institute"

Steve Vu

# 1. Introduction
The project aims at building a simulator that trains a lunar module, thereafter known as an agent, to land on the moon. We frame it as a reinforcement learning problem that the agent learns simply through interaction with the dynamics of the world. The ultimate goal is to find a robust and efficient landing policy for the agent.

As the project simplifies the complexities of outer space, it is a great starting point to gain experience in converting word descriptions of problems into a concrete solution.

# 2. Implementation

## 2.1 Create an environment
The project starts with creating a lunar lander environment that is well developed by [OpenAI](https://github.com/openai/gym). The goal is to land the lunar module in the landing zone located between the two yellow flags. The landing zone is in the same location, but the shape of the ground around it may change. The agent can fire the main thruster or either of the side thrusters to orient the module and slow its descent.

The state is composed of eight variables. It includes the XY position, the XY velocity, the angle, the angular velocity and the signals from each leg’s sensor that determines if the module is touching the ground.

There are four possible actions as the agent can fire the main thruster, fire the left thruster, fire the right thruster, or do nothing on this step.
About the reward function, we encourage the agent to move towards the goal, pile to the surface in a safe way, use fuel efficiently, and not fly off into outer space.

https://user-images.githubusercontent.com/83741887/127781646-e6242455-dd4f-43f5-b1af-40a64917b07b.mp4

## 2.2 Choose the learning algorithm
While the RL textbook, <b>Reinforcement Learning: An Introduction by Sutton and Barto</b>, and the RL Specialization provides informative surveys of RL algorithms, we need to find a well-fitted algorithm for the problem. By the end, <b>the Expected SARSA</b> is able to outperform other algorithms in the specific problem.

The Expected SARSA is to deal with continuous state variables and episodic learning process of the problem. It is also beneficial as we are able to update the policy and value function on every time step before the end of the episode with the algorithm. Finally, the Expected SARSA allows us to learn an optimal epsilon soft policy that is more robust than learning a deterministic policy.

![](https://github.com/SteveVu2212/LandingOnTheMoon/blob/main/Images/Learning%20Algorithm.png)

## 2.3 Make design choices
Which function approximation will be employed is one of the most important decision as the number of features grows exponentially with the input dimension. Using neural network is the best choice for the problem. Besides setting the neural network’s parameters, picking an effective activation function is critical. By the end, we go ahead with <b>ReLUs function</b> as it avoids the issue of saturation in neural networks.

![](https://github.com/SteveVu2212/LandingOnTheMoon/blob/main/Images/ReLUs%20activation%20func.png)

Training the neural network is not easy as stochastic gradient descent is too slow. We will learn the advantage of the <b>ADAM optimizer</b> which combines adaptive vector stepsizes and momentum to speed up learning. The weights are updated as follows:

![](https://github.com/SteveVu2212/LandingOnTheMoon/blob/main/Images/ADAM%20update.png)

where <img src="https://render.githubusercontent.com/render/math?math=$\mathbf{\hat{m}}$"> and <img src="https://render.githubusercontent.com/render/math?math=$\mathbf{\hat{v}}$"> are the unbiased estimates of the mean and second moment, <img src="https://render.githubusercontent.com/render/math?math=$\mathbf{w}$"> and <img src="https://render.githubusercontent.com/render/math?math=$\mathbf{s}$">, which are initialized to zero.

The exploration – exploitation tradeoff is a fundamental problem in Reinforcement Learning. While it is difficult to maintain the effect of optimistic initial values on exploration when using neural networks, the epsilon greedy approach has a downside of ignoring the information about the action values. <b>Softmax policy</b> is the best choice as the probability of selecting an action is proportional to the value of the action. Notably, we subtract the maximum action value from the action values to avoid overflow.

![](https://github.com/SteveVu2212/LandingOnTheMoon/blob/main/Images/Softmax%20funcs.png)

Learning the idea of planning from Dyna-Q that uses simulated experience to improve the value estimates, we would like to employ the <b>experience replay</b> method to make the agent more sample efficient when using function approximation. We simply save a buffer of experience and let the data be the model before using several samples from the buffer, called a mini batch, and updating the value function with those samples.

# 3. Prerequisites
It is highly recommended that you complete [Course 1: Fundamentals of Reinforcement Learning](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning), [Course 2: Sample-Based Learning Methods](https://www.coursera.org/learn/sample-based-learning-methods) and [Course 3: Prediction and Control with Function Approximation](https://www.coursera.org/learn/prediction-control-function-approximation) before starting the project.

