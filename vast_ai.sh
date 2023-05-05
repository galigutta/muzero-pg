git clone https://github.com/galigutta/muzero-pg.git
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
./Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
conda create -n muzr python=3.9
conda activate muzr
cd muzero-pg
pip install -r requirements.lock
pip install "ray[default]"
pip install aiohttp==3.7.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118