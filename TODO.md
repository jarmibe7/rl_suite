Logistic
[] Need to figure out how to deal with terminated/truncated being saved into the "done" category in the replay buffer. SAC only uses terminated, but other algs might use truncated.
[] Create a policy playback script that can play and eval policy checkpoints from run directory
[] Need to fix sequence mode sampling in replay buffer. Currently it saves entire episodes, selects random random episodes without replacement, and then samples a random sample of sequence_length from each episode. Instead, I want it to work more similarly to transition mode. One option is it could accumulate a trajectory of sequence_length before adding to buffer, and another option is that it could just sample sequence_length transitions that don't wrap around to a new epsiode (check terminated flag).

Algorithms
[] World model actor-critic method that is basically SAC but takes in the latent state inside the world model. May even be able to use the base SAC algorithm with RSSM wrapper...