# MEU6505 Project 6: Hierarchical LLM-guided Symmetry-aware Bimanual Manipulation

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](https://opensource.org/license/apache-2-0)

> 연세대학교 기계공학과 MEU6505 최적제어 및 강화학습 수업 프로젝트 6  
> Optimal Control and Reinforcement Learning Course - Project 6

<p align="center">
  <img src="https://github.com/user-attachments/assets/88dfa8d7-1f10-4c78-999b-fc86a6b95b13" alt="GR1T2 Bimanual Manipulation" width="800"/>
</p>

## 📖 Overview

**Groot**는 대형 언어 모델(LLM)을 활용하여 복잡한 양팔 조작(bimanual manipulation) 작업을 계층적으로 해결하는 프레임워크입니다. LLM이 대칭성을 고려한 하위 목표(subgoal)를 생성하면, 강화학습(RL) 컨트롤러가 이를 실행합니다.

**Groot** is a hierarchical framework that uses Large Language Models (LLMs) to generate symmetry-aware subgoals and RL controllers to execute them for efficient bimanual manipulation tasks in simulation. Built on Isaac Lab, it targets long-horizon, high-dimensional robotic tasks.

### 🎯 Key Features

- **계층적 제어 구조**: LLM 기반 고수준 계획 + RL 기반 저수준 실행
- **대칭성 인식**: 양팔 조작에서 대칭성을 활용한 효율적인 학습
- **Isaac Lab 기반**: NVIDIA Isaac Sim의 고성능 시뮬레이션 환경
- **유연한 확장성**: 다양한 로봇과 작업에 적용 가능

### 🤖 Motivation and Background

- 장기적인 양팔 조작 작업(예: 천 접기, 병뚜껑 열기)은 큰 상태-행동 공간으로 인해 일반적인 강화학습으로 해결하기 어렵습니다
- 기존 LLM 기반 플래너(LABOR, LLM+MAP 등)는 스크립트 기반 스킬에 의존하여 유연성과 일반화에 제한이 있습니다
- 본 프로젝트는 LLM이 생성한 대칭성 인식 하위 목표를 RL 컨트롤러가 실행하는 계층적 프레임워크를 개발합니다

### 📊 Dataset and Environment

- Isaac Lab의 시뮬레이션 양팔 조작 작업 환경 활용
- **입력**: 휴머노이드 관절 구성 및 객체 위치 정보
- **출력**: RL 컨트롤러가 실행할 하위 목표

---

## 🚀 Quick Start

### Prerequisites

- Ubuntu 22.04 LTS
- Python 3.10 이상
- NVIDIA GPU (RTX 시리즈 권장)
- CUDA 11.8 이상

### Installation

**Step 1.** Isaac Lab 설치

Isaac Lab 공식 설치 가이드를 따라주세요: [Installation Guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)

**Step 2.** 레포지토리 클론

```bash
git clone --recursive https://github.com/dddongri/MEU6505_Project_6.git
cd MEU6505_Project_6
```

**Step 3.** Groot 라이브러리 설치

Isaac Lab이 설치된 Python 환경에서 다음 명령어를 실행합니다:

```bash
cd source/groot
python -m pip install -e .
```

**Step 4.** (선택사항) Pre-commit 훅 설치

코드 포맷팅과 린팅을 자동화하려면:

```bash
pip install pre-commit
pre-commit install
```

---

## 💻 Usage

### 환경 목록 확인

사용 가능한 모든 환경을 확인합니다:

```bash
python scripts/list_envs.py
```

**사용 가능한 작업 (Available Tasks):**
- `Template-Isaac-Velocity-Flat-Anymal-D-v0` - Anymal-D 로봇 평지 이동
- `Template-Isaac-Velocity-Rough-Anymal-D-v0` - Anymal-D 로봇 험지 이동
- **`GR1T2-Basic`** - GR1T2 휴머노이드 기본 양팔 조작
- **`GR1T2-PickPlace`** - GR1T2 픽앤플레이스 작업
- **`GR1T2-HandToHand`** - GR1T2 손-손 객체 전달 작업

### 학습 (Training)

특정 작업에 대해 강화학습 학습을 시작합니다:

```bash
# 기본 학습
python scripts/rsl_rl/train.py --task TASK_NAME

# Headless 모드 (GUI 없이)
python scripts/rsl_rl/train.py --task TASK_NAME --headless

# 예시: GR1T2 픽앤플레이스 작업 학습
python scripts/rsl_rl/train.py --task GR1T2-PickPlace --headless
```

**주요 학습 옵션:**
- `--task`: 학습할 작업 이름
- `--headless`: GUI 없이 학습 (서버 환경에서 권장)
- `--num_envs`: 병렬 환경 개수 (기본값: 작업별로 상이)

> **💡 Tip:** Groot는 `rsl_rl`, `Stable Baselines3` 및 커스텀 강화학습 알고리즘과 호환됩니다.

### 평가 (Evaluation)

학습된 모델을 평가합니다:

```bash
# 최신 로그 디렉토리 자동 로드
python scripts/rsl_rl/play.py --task TASK_NAME

# 특정 로그 디렉토리 지정
python scripts/rsl_rl/play.py --task TASK_NAME --log_dir PATH_TO_LOG --num_envs NUM_ENVS

# 예시
python scripts/rsl_rl/play.py --task GR1T2-PickPlace --num_envs 16
```

> **💡 Tip:** `--log_dir`을 지정하지 않으면 자동으로 최신 로그가 로드됩니다!

### 학습 모니터링

TensorBoard를 통해 실시간으로 학습 로그를 모니터링할 수 있습니다:

```bash
tensorboard --logdir logs/rsl_rl/TASK_NAME/
```

브라우저에서 `http://localhost:6006`으로 접속하여 학습 진행 상황을 확인하세요.

---

## 📁 Project Structure

```
MEU6505_Project_6/
├── source/groot/              # Groot 라이브러리 메인 코드
│   ├── groot/
│   │   └── tasks/            # 작업 정의
│   │       ├── locomotion/   # 이동 작업 (Anymal-D)
│   │       └── manipulation/ # 조작 작업 (GR1T2)
│   │           ├── pick_place/      # 픽앤플레이스
│   │           └── hand_to_hand/    # 손-손 전달
│   ├── config/               # 확장 설정
│   └── setup.py              # 설치 스크립트
├── scripts/
│   ├── rsl_rl/              # RSL-RL 학습/평가 스크립트
│   │   ├── train.py         # 학습 스크립트
│   │   └── play.py          # 평가 스크립트
│   ├── list_envs.py         # 환경 목록 출력
│   └── rename_template.py   # 템플릿 이름 변경
├── docs/                     # 문서 및 이미지
├── logs/                     # 학습 로그 (생성됨)
├── README.md                 # 본 파일
└── README_setup.md          # 상세 설정 가이드
```

---

## 🛠️ Code Formatting

본 프로젝트는 pre-commit을 사용하여 코드 스타일을 자동으로 관리합니다.

### Pre-commit 설치 및 사용

```bash
# Pre-commit 설치
pip install pre-commit

# Pre-commit 훅 설치
pre-commit install

# 모든 파일에 대해 수동 실행
pre-commit run --all-files
```

커밋 시 자동으로 코드 포맷팅과 린팅이 수행됩니다.

---

## 📚 References

### 관련 논문
- **LABOR**: [Learning to Acquire Novel Bimanual Object Manipulation through Large Language Models](https://arxiv.org/pdf/2404.02018)
- **Isaac Lab**: [Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

### 사용 기술
- [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) - 고성능 로봇 시뮬레이션
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab) - 강화학습 환경 프레임워크
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - 강화학습 라이브러리

---

## 👥 Contributors

- **Author**: [Sol Choi](https://github.com/S-CHOI-S) (최솔)
- **Course**: MEU6505 - Optimal Control and Reinforcement Learning
- **Institution**: Yonsei University, Department of Mechanical Engineering

---

## 📄 License

이 프로젝트는 Apache License 2.0 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENCE) 파일을 참조하세요.

```
Copyright 2024 The Isaac Lab Project Developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
```

---

## 🙏 Acknowledgments

이 프로젝트는 다음을 기반으로 개발되었습니다:
- [Isaac Lab Extension Template](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- NVIDIA Isaac Lab 프레임워크
- 연세대학교 MEU6505 수업

---

## 📞 Contact

프로젝트에 대한 문의사항이나 이슈가 있으시면:
- GitHub Issues: [Create an issue](https://github.com/dddongri/MEU6505_Project_6/issues)
- Author: [@S-CHOI-S](https://github.com/S-CHOI-S)

---

<p align="center">
  Made with ❤️ for MEU6505 Optimal Control and Reinforcement Learning
</p>
