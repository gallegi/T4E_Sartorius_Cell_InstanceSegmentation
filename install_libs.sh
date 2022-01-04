# 1. install pytorch, you might have to change the above line to make it compatible with your cuda version
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# 2. install detectron2
# python -m pip install 'git+https://github.com/zhanghang1989/detectron2-ResNeSt.git'
pip install --upgrade pyyaml==5.1
python -m pip install  --upgrade detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# 3. install other dependencies
pip install -r requirements.txt

# 4. Make detectron2 works with ResNeSt200
python add_resnest2.py