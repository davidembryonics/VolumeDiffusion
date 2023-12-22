pip install -r requirements.txt
git clone https://github.com/threestudio-project/threestudio.git
cd ./threestudio
git reset --hard 3fe3153bf29927459b5ad5cc98d955d9b4c51ba3
cp ../refine/networks.py ./threestudio/models/
cp ../refine/base.py ./threestudio/models/prompt_processors/
pip install -r requirements.txt