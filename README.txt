Due to the IP condential of the simulation toolbox utilized in this project, it was agreed with the project supervisor that the project code shall not be uploaded at this stage.

However, some description about the work that has been done are listed as below:
1. Configurations of the simulator, enabling it to run with sample scenarios and customized scenarios;
2. Updates of the Observation.py function in a gym environment
2.1 adding a new class of observation
2.2 adding functions to generated required grid map for the agent to observe
2.3 data process to return the allowed format of information
2.4 change of environment setting for new observations
3. Updates of the Reward.py function in a gym environment
3.1 adding a new class of observation
3.2 adding functions to calculate the required rewards
4. codes for running the Deep Reinforcement Learning Agent with stable baseline3
4.1 a script that initialize the environment, and perform a simulation with arbitary controls of ships
4.2 add Minitor functions to stor the training process
4.3 add an image store and video generation function to visualize the sychronous grid map of simulator 
4.4 create a customized PPO agent, with CnnPolicy
4.5 training of the agent with modified gym environment
4.6 store the trainig process, and visualize the results.
4.7 validation simulation with trained agent that perform collision avoidance tasks in the scenario
5. version control and documentation of the codes. 