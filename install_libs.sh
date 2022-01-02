# 1. install pytorch, you might have to change the above line to make it compatible with your cuda version
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# 2. install detectron2
python -m pip install 'git+https://github.com/zhanghang1989/detectron2-ResNeSt.git'
# 3. install other dependencies
pip install -r requirements.txt
