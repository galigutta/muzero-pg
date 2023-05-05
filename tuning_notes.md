Tuning Notes
1. Tried with a super large batch (512) and a lot of selfplay games (24). Convergence is very slow. Fluctuating wildly after about 20k games. step speed on A6000 is about half of what was observed on a 4090. (on Vast.ai)
2. Adam optimizer is way better than SGD with momentum for degentrader. 
3. Tuned visit softmax temperature to decay over time
4. Running the same model on the same gpu with the same see gives models that converge to different policies. 
5. Need to checkpoint / save models (effectively early stopping)
6. Think a little about curriculum learning. Train on easier steps first and present complex steps later.
7. For the degentrader problem, it trains very quickly with fully connected network with few parameters.
8. Keep in mind the relationship between replay buffer size, batch size (currently 64) and the number of games on reanalyze. Dont want the buffer to be too big as it reduces the effectiveness of reanalyze. Dont want it to be too small as it misses out on varied experiences (?)
9. Need to understand what replay buffer does. It seems to store all the actual games to disk.

