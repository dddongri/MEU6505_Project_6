# rsl_rl_ppo_cfg.py (요지)
algo:
  name: "RSL_RL_PPO"
  num_steps: 24
  horizon_length: 24
  clip_coef: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coef: 0.0
  value_coef: 2.0
  lr: 3e-4
  max_grad_norm: 1.0
policy:
  actor_hidden_dims: [256, 256]
  critic_hidden_dims: [256, 256]
  activation: "elu"
