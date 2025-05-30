# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import subprocess
import sys
import time
import signal
import os

# List of MCP servers to run
# 실행할 MCP 서버 목록
mcp_servers = [
    "application/mcp_server_tavily.py",
    "application/mcp_server_arxiv.py",
    "application/mcp_server_pubmed.py",
    "application/mcp_server_chembl.py",
    "application/mcp_server_clinicaltrial.py",
]

processes = []

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Shutting down all servers...")
    for process in processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("Starting all MCP servers...")
    
    for server in mcp_servers:
        print(f"Starting {server}...")
        
        # Validate server path before execution
        if not os.path.isfile(server) or not server.startswith("application/mcp_server_"):
            print(f"Error: Invalid server path {server}")
            continue
            
        # Start server with validated path
        # 서버 시작
        process = subprocess.Popen(
            [sys.executable, server],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        processes.append(process)
        
        # Small delay until server starts
        # 서버가 시작될 때까지 약간의 지연
        time.sleep(1)
        
        # Check initial log output
        # 초기 로그 출력 확인
        if process.poll() is not None:
            # If process has already terminated
            stdout, stderr = process.communicate()
            print(f"Error: Failed to start {server}")
            print(f"STDERR: {stderr}")
            print(f"STDOUT: {stdout}")
            # Terminate all other processes
            # 다른 모든 프로세스 종료
            for p in processes:
                if p != process and p.poll() is None:
                    p.terminate()
            sys.exit(1)
    
    print("\nAll MCP servers started successfully.")
    print("Server logs:")
    
    # Monitor logs of all servers in real-time
    # 모든 서버의 로그를 실시간으로 모니터링
    try:
        while True:
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    # If process terminated unexpectedly
                    # 프로세스가 예기치 않게 종료된 경우
                    stdout, stderr = process.communicate()
                    print(f"\nError: {mcp_servers[i]} server terminated unexpectedly.")
                    print(f"STDERR: {stderr}")
                    # Terminate all other processes
                    # 다른 모든 프로세스 종료
                    for p in processes:
                        if p != process and p.poll() is None:
                            p.terminate()
                    sys.exit(1)
                
                # Read standard output and errors
                # 표준 출력 및 오류 읽기
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    print(f"[{mcp_servers[i]}] {line.strip()}")
                
                for line in iter(process.stderr.readline, ""):
                    if not line:
                        break
                    print(f"[{mcp_servers[i]} ERROR] {line.strip()}")
                    
            time.sleep(0.1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()