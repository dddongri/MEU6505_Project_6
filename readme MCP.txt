1. 클로드 앱 설치

2. mcp server 구동

uv run --directory ~/YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp ~/YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp/isaac_mcp/server.py

3. 클로드 앱에서 settings -> developer -> edit config -> claude_desktop_config.json 파일에 서버 정보 입력
(경로 입력시 절대경로 사용 권장)

"""
{
  "mcpServers": {
    "mcp-server-omni-isaacsim": {
        "type": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--directory",
            "/YOUR_PATH/MEU6505_Project_6/omni-mcp",
            "/YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp/isaac_mcp/server.py"
        ]
    }
  }
}

"""

4. 시뮬레이터 실행
python scripts/rsl_rl/run.py --task GR1T2-HandToHand --num_envs 1 --kit_args "--ext-folder /home/qkd/Desktop/MEU6505_Project_6/omni-mcp/isaac-sim-mcp/ --enable isaac.sim.mcp_extension --enable omni.isaac.nucleus"




