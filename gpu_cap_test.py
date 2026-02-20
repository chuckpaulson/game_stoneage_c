import torch, time, numpy as np
from stone_age_net import StoneAgeNN, OBS_SIZE

nn = StoneAgeNN(num_players=2, hidden=256, blocks=3)
for bs in [1, 8, 16, 32, 64, 128, 256]:
    obs = np.random.randint(0, 10, (bs, OBS_SIZE), dtype=np.int32)
    # warmup
    for _ in range(10):
        nn.predict_batch(obs)
    t0 = time.time()
    for _ in range(200):
        nn.predict_batch(obs)
    dt = time.time() - t0
    print(f"  batch={bs:4d}:  {dt/200*1000:.2f} ms total,  {dt/200/bs*1000*1000:.0f} µs/sample")