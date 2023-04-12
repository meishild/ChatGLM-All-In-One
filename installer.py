import shutil
import sys
import subprocess
import os
import pkg_resources
from zipfile import ZipFile

try:
    import requests
except ModuleNotFoundError as error:
    required = {'requests'}
    installed = {p.key for p in pkg_resources.working_set}
    missing = required - installed
    if missing:
        print("installing requests...")
        python = sys.executable
        subprocess.check_call([python, "-m", "pip", "install", *missing], stdout=subprocess.DEVNULL)


def main():
    if sys.platform != "win32":
        print("ERROR: This installer only works on Windows")
        quit()
    else:
        print("Running on windows...")

    version = sys.version_info
    if version.major != 3 or version.minor != 10 or version.micro < 6:
        print("ERROR: You don't have python version than 3.10.6 installed, please install python 3.10.6, and add it to path")
        quit()
    else:
        print("Python version than 3.10.6 detected...")

    python_real = sys.executable
    python = r"venv\Scripts\pip.exe"

    print("creating venv and installing requirements")
    subprocess.check_call([python_real, "-m", "venv", "venv"])

    reply = None
    while reply not in ("0", "1", "2"):
        reply = input("which version of torch do you want to install?\n"
                      "0 = 1.12.1\n"
                      "1 = 2.0.0\n"
                      "2 = 2.1.0: ").casefold()

    if reply == "2":
        torch_version = "torch==2.1.0.dev20230322+cu118 torchvision==0.16.0.dev20230322+cu118 --extra-index-url https://download.pytorch.org/whl/nightly/cu118"
    elif reply == '1':
        torch_version = "torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_version = "torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
    print("installing torch")
    subprocess.check_call(f"{python} install {torch_version}".split(" "))

    print("installing other requirements")
    subprocess.check_call(f"{python} install -r requirements.txt".split(" "))

    print("Setting up default config of accelerate")
    with open("default_config.yaml", 'w') as f:
        f.write("command_file: null\n")
        f.write("commands: null\n")
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("deepspeed_config: {}\n")
        f.write("distributed_type: 'NO'\n")
        f.write("downcase_fp16: 'NO'\n")
        f.write("dynamo_backend: 'NO'\n")
        f.write("fsdp_config: {}\n")
        f.write("gpu_ids: '0'\n")
        f.write("machine_rank: 0\n")
        f.write("main_process_ip: null\n")
        f.write("main_process_port: null\n")
        f.write("main_training_function: main\n")
        f.write("megatron_lm_config: {}\n")
        f.write("mixed_precision: fp16\n")
        f.write("num_machines: 1\n")
        f.write("num_processes: 1\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")
        f.write("tpu_name: null\n")
        f.write("tpu_zone: null\n")
        f.write("use_cpu: false")
    if os.path.exists(os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface',
                                   'accelerate', 'default_config.yaml')):
        os.remove(os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface', 'accelerate', 'default_config.yaml'))
    shutil.move("default_config.yaml", os.path.join(os.environ['USERPROFILE'], ".cache", "huggingface", "accelerate"))

    for file in os.listdir("bitsandbytes_windows"):
        shutil.copy(os.path.join('bitsandbytes_windows', file),
                    os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes'))
    shutil.copy(os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'main.py'),
                os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'cuda_setup'))

    reply = None
    while reply not in ("y", "n"):
        reply = input(f"Do you want to install the optional cudnn patch for faster "
                      f"training on high end 30X0 and 40X0 cards? (y/n): ").casefold()

    if reply == 'y':
        r = requests.get("https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip")
        with open("cudnn.zip", 'wb') as f:
            f.write(r.content)
        with ZipFile("cudnn.zip", 'r') as f:
            f.extractall(path=".")
        subprocess.check_call(f"{os.path.join('venv', 'Scripts', 'python.exe')} {'cudnn.py'}".split(" "))
        shutil.rmtree("cudnn_windows")
        shutil.rmtree("cudnn.zip")
    else:
        reply = None
        while reply not in ('y', 'n'):
            reply = input("Are you using a 10X0 series card? (y/n): ")
        if reply:
            shutil.copy(os.path.join("..", "installables", "libbitsandbytes_cudaall.dll"),
                        os.path.join("venv", 'Lib', 'site-packages', 'bitsandbytes'))
            os.remove(os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'cuda_setup', 'main.py'))
            shutil.copy(os.path.join('..', 'installables', 'main.py'),
                        os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'cuda_setup'))
    print("Completed installing.")


if __name__ == "__main__":
    main()
