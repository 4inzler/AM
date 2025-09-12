import subprocess
import os
import shutil
import sys

def run_cmd(cmd, check=True):
    print(f"➤ {' '.join(cmd)}")
    subprocess.run(cmd, check=check)

def install_sherlock():
    sherlock_dir = os.path.expanduser("~/sherlock")
    sherlock_script = os.path.join(sherlock_dir, "sherlock")
    sherlock_link = "/usr/local/bin/sherlock"

    # === Step 1: Clone the repo
    if not os.path.exists(sherlock_dir):
        print("📦 Cloning Sherlock repository...")
        run_cmd(["git", "clone", "https://github.com/sherlock-project/sherlock.git", sherlock_dir])
    else:
        print("✔️ Sherlock already cloned.")

    # === Step 2: Install using uv or pip
    os.chdir(sherlock_dir)
    if shutil.which("uv"):
        print("🚀 Detected uv — installing with uv...")
        run_cmd(["uv", "pip", "install", "."])
    else:
        print("🐍 Using pip to install pyproject.toml project...")
        run_cmd([sys.executable, "-m", "pip", "install", "."])

    # === Step 3: Symlink if needed
    if not os.path.exists(sherlock_link):
        print("🔗 Linking sherlock to /usr/local/bin...")
        run_cmd(["sudo", "ln", "-s", sherlock_script, sherlock_link])
    else:
        print("✔️ Sherlock already linked at /usr/local/bin.")

    print("\n✅ Sherlock is installed and ready to serve AM.")

if __name__ == "__main__":
    try:
        install_sherlock()
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
