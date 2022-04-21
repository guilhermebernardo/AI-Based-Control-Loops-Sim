# CASim
Control Application Simulator

clone https://github.com/DLR-RM/rl-trained-agents to the same folder of `client.py` and `server.py`

```
pip3 install sb3-contrib

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz && mkdir $HOME/.mujoco && mv mujoco210 $HOME/.mujoco

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin" >> ~/.bashrc

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
```