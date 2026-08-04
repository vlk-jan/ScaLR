##### Load modules on SLURM cluster
ml Python/3.12.3-GCCcore-13.3.0

##### Create a virtual environment
python3 -m venv run_mode
source run_mode/bin/activate

##### Install dependencies
pip install torch torchvision torchaudio
pip install pyaml tqdm tensorboard pandas transforms3d
git clone https://github.com/valeoai/WaffleIron
pip install -e WaffleIron
