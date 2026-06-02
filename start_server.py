"""Start the uvicorn server as a detached process on Windows."""
import subprocess
import sys
import os

# Get the project root
project_root = os.path.dirname(os.path.abspath(__file__))
python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")
uvicorn_module = "uvicorn"

# Start as a detached process (doesn't die when this script ends)
proc = subprocess.Popen(
    [python_exe, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000",
     "--host", "0.0.0.0"],
    cwd=project_root,
    stdout=open(os.path.join(project_root, "server.log"), "w"),
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    if sys.platform == "win32" else 0,
)
print(f"Server started with PID {proc.pid}", flush=True)
