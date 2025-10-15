## Setup

### No gpu
1. Create venv
2. run "pip install -r requirements.txt"

### Gpu
1. Create venv
2. Install cuda drivers - https://developer.nvidia.com/cuda-downloads
3. run command to install pytorch with cuda support, find the right version and command here - https://pytorch.org/get-started/locally/
4. first run your pytorch install command e.g. "pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130"
5. lastly run "pip install -r requirements.txt"


## Structure
1. Base model stores basic code for training a deep Q learning model
2. Models folder stores trained models ready for evaluation
3. Test Script is used for running evaluation on models