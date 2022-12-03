# Code for the 4th Solution (Team Durrett) at Kaggle LuxAI 2021
## Hardware
- CPU specs: Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz
- number of CPU cores: 8
- GPU specs: GeForce GTX 1080Ti
- number of GPUs: 2

## OS/platform
- Ubuntu20.04 (512GB boot disk)
- 48GB memory

## 3rd-party software
- Docker 20.10.11
Dockerfile we used is [Kaggle/docker-python](https://github.com/Kaggle/docker-python). 
  
## train model
```
python train.py
```

## prediction
If there exists model.pth, you can use agent.py.

## Configuration
If you want to change the default training parameters or data-path, you can rewrite the config.py.

## Note
- If you run train.py and finish training, model.pth will be generated, then the original model.pth will be overwritten.