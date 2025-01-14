#!/bin/bash
JUPYTER_PASSWORD="chemCPA"

# Source conda
source /home/user/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate chemCPA

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/user/conda/envs/chemCPA/lib:$LD_LIBRARY_PATH

# Create jupyter config directory if it doesn't exist
mkdir -p ~/.jupyter

# Set up the password
python -c "from jupyter_server.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))" > ~/.jupyter/jupyter_server_password.txt
HASHED_PASSWORD=$(cat ~/.jupyter/jupyter_server_password.txt)

# Create jupyter server config
cat > ~/.jupyter/jupyter_server_config.py << EOF
c.ServerApp.password = '$HASHED_PASSWORD'
c.ServerApp.password_required = True
EOF

chmod 1777 /tmp
chmod 755 /home/user

sudo apt update;
DEBIAN_FRONTEND=noninteractive  sudo apt-get install openssh-server -y;
mkdir -p ~/.ssh;
cd $_;
chmod 700 ~/.ssh;
echo "$PUBLIC_KEY" >> authorized_keys;
chmod 700 authorized_keys;
service ssh start;

echo "Starting Jupyter Lab with password authentication"

NOTEBOOK_DIR="/home/user/chemCPA"

jupyter-lab \
    --ip 0.0.0.0 \
    --port 8888 \
    --no-browser \
    --allow-root \
    --notebook-dir=$NOTEBOOK_DIR
