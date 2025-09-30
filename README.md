## MEU6505_Project_6


Optimal control and reinforcement learning class project 6 


<img width="1099" height="619" alt="image" src="https://github.com/user-attachments/assets/88dfa8d7-1f10-4c78-999b-fc86a6b95b13" />

## 자료
LABOR paper: https://arxiv.org/pdf/2404.02018
Isaac Sim: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html


## GPT가 적으라 한 거
git clone git@github.com:dddongri/MEU6505_Project_6.git

cd PROJECT

uv sync                      # 또는 conda/poetry

pre-commit install           # 커밋 전 자동 포맷/린트

pytest -q                    # 최소 테스트 통과 확인

python experiments/run_ppo.py +env=cartpole seed=0
