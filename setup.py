import platform
import subprocess

PYTHON = "py -3.12"  # <--- Update here

def run(command):
    subprocess.run(command, shell=True, check=True)

is_windows = platform.system() == "Windows"
venv_dir = ".venv"

# Create venv
run(f"{PYTHON} -m venv {venv_dir}")
activate = f"{venv_dir}\\Scripts\\activate.bat" if is_windows else f"source {venv_dir}/bin/activate"
print(f"Activate with: {activate}")

# Install deps
python_exec = f"{venv_dir}\\Scripts\\python" if is_windows else f"{venv_dir}/bin/python"
run(f"{python_exec} -m pip install --upgrade pip")
run(f"{python_exec} -m pip install -r requirements.txt")