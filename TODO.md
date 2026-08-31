Logistic
[] Need to figure out how to deal with terminated/truncated being saved into the "done" category in the replay buffer. SAC only uses terminated, but other algs might use truncated.
[] Create a policy playback script that can play and eval policy checkpoints from run directory

Algorithms
[] World model actor-critic method that is basically SAC but takes in the latent state inside the world model. May even be able to use the base SAC algorithm with RSSM wrapper...