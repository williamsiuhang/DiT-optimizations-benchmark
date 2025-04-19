# Inference Optimization Benchmark

## Installation

### On remote ubuntu / linux

1. Manually intall rsync on ubuntu: `apt update && apt install rsync`

2. Create .env with your variables to be used in syncing scripts

```bash
PORT          = 22                                  # Port for SSH connection
SSH_KEY_PATH  = ~/.ssh/ssh_for_instance_connection  # Path to SSH private key
USER          = root                                # SSH username
IP            = 12.12.123                           # Remote server IP address
TARGET_VOLUME = /workspace/                         # Directory on the remote server
```

3. Sync project content to remote: `bash scripts/setup/sync_remote.sh`

4. SSH into instance, then run setup scripts

`bash scripts/setup/setup1.sh` _installs miniconda_

`source ~/.bashrc` _restart shell so miniconda works_

`conda deactivate` _if it's in (base) after restarting_

`bash scripts/setup/setup2.sh` _set up environment, download pytorch and datasets_

### On local machine

Create conda environment

`conda env create --file environment.yml`

## Sync local project directory to remote

`bash scripts/setup/sync_remote.sh`
