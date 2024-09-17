ls
cd ./mmgdino
cd ./mm-gdino
cd ..
git clone git@192.168.100.15:ModelTrains/mm-grounddino.git
conda env list
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh -O anaconda.sh
ls
cd ./Clash
ls
./clash -d . > 20240520.log
sudo docker login 192.168.100.25:80 -u admin -p Harbor-+12345
tar -czvf mmgdino.tar.gz -C /home/cxc/mm-gdino
ls -l /home/cxc/mm-gdino
tar -czvf mmgdino.tar.gz -C /home/cxc mm-gdino
ls
docker exec -it mm-grounding-dino-container /bin/bash
docker exec -it mm-grounding-dino- /bin/bash
docker exec -it mm-grounding-dino /bin/bash
sudo docker run -it --rm     -p 5678:5678     -v $(pwd):/home/cxc/mm-gdino     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --name mm-grounding-dino     192.168.100.25/algolib/mm-grounding-dino:demo
docker run -it --rm     -p 5678:5678     -v $(pwd):/home/cxc/mm-gdino     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --name mm-grounding-dino     192.168.100.25/algolib/mm-grounding-dino:demo
ssh cxc@192.168.100.25
sudo docker login 192.168.100.25:80 -u admin -p Harbor-+12345
sudo docker login 192.168.100.25 -u admin -p Harbor-+12345
docker images
sudo docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
git --version
git init
git config --global user.name "cxc"
git config --global user.email "Cxc65000@gmail.com"
git config --global core.editor vim
git add.
git add .
git add ./configs/mm_grounding_dino/grounding_dino_sein-l_finetune_smokefire_test.py
git commit -m "add a new feature"
vi /etc/ssh/sshd-config
ssh-keygen -t rsa -b 4096 -C "cxc65000@gmail.com"
ssh-copy-id git@192.168.100.15
scp -r ./* cxc@192.168.2.46: /home/cxc
scp -r ./* cxc@192.168.2.46:/home/cxc
scp -r ./* cxc@192.168.2.46:/home/cxc/mm-gdino
scp -r ./data/smoke_fire cxc@192.168.2.46:/home/cxc/mm-gdino/data
git config --global user.name "caixichen"
git config --global user.email "caixichen@yourangroup.com"
ssh-keygen -t rsa -b 4096 -C "caixichen@yourangroup.com"
ls
cd /home/cxc/.ssh/id_rsa
cd /home/cxc/.ssh
ls
vi id_rsa.pub
docker ps
docker stop 391fdc6cf17e
docker rm 391fdc6cf17e
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker exec -it 947935d8f355 /bin/bash
ssh root@127.0.0.1 -p 8043
docker exec -it 947935d8f355 /bin/bash
docker ps
docker stop 0f5c388882e8
docker rm 0f5c388882e8
docker exec -it 391fdc6cf17e /bin/bash
ssh root@127.0.0.1 -p 8022
docker exec -it 391fdc6cf17e /bin/bash
ssh root@127.0.0.1 -p 8043
vi ~/.ssh/known_hosts
ssh-keygen -f "/home/cxc/.ssh/known_hosts" -R "[127.0.0.1]:8043"
docker exec -it 0f5c388882e8 /bin/bash
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
nmap 192.168.100.25
docker ps
docker exec -it 391fdc6cf17e /bin/bash
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker ps
docker stop d61ee979d1d7
docker rm d61ee979d1d7
docker ps
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker ps
docker exec it 0f5c388882e8 /bin/bash
docker exec -it 0f5c388882e8 /bin/bash
python
phython3
pip
docker exec -it d61ee979d1d7 /bin/bash
docker ps
docker a
docker ps
docker stop 6dac29498f3a
docker mv 6dac29498f3a
docker rm 6dac29498f3a
docker ps
docker images
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker ps
docker exec -it d61ee979d1d7 /bin/bash
ssh root@127.0.0.1 -p 8043
docker run -it >     -v /tmp/.X11-unix:/tmp/.X11-unix >     -e DISPLAY=$DISPLAY >     -v /etc/timezone:/etc/timezone >     -v /home:/home >     -v /home/cxc:/home/cxc >     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility >     --shm-size 32G >     --gpus all >     --cap-add=SYS_PTRACE >     --security-opt seccomp=unconfined >     192.168.100.25/algolib/mm-grounding-dino:demo
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /h
ome/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt 
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /
ome/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt 
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
docker exec -it -w /home/cxc/mm-gdino mm-grounding-dino-container /bin/bash
docker ps
docker exec -it -w /home/cxc/mm-gdino quirky_jang /bin/bash
docker exec -it 72121b419070  -w /home/cxc/mm-gdino quirky_jang /bin/bash
docker exec -it 72121b419070  /bin/bash
docker exec -it -w /home/cxc/mm-gdino 72121b419070 /bin/bash
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8022:22     -p 8081:8080     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8022:22     -p 4041:4040     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
docker kill 3c1aa4853ac2
docker ps -f "publish=8022"
docker ps -f "publish=8043"
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -p 4041:4040     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
netstat -ntulp | grep 4041
apt install net-tools
sudo apt install net-tools
docker ps -a
docker rm 1fb4fb7d7d82
docker rm 9e9bd020c48e
docker rm 1a2638e825a4
docker rm 64815aec8279
docker rm 1fb4fb7d7d82
docker ps -a
docker rm c40580de7b5e
docker stop 72121b419070
docker rm 72121b419070
sudo docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --name mm-grounding-dino     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --name mm-grounding-dino     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker ps -f "publish=8043"
docker stop c40580de7b5e
docker rm c40580de7b5e
docker ps -f "publish=8043"
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --name mm-grounding-dino     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8043:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker ps
docker exec -it 6dac29498f3a87cfaa66ca976b7d43e11a2cf435c44786c52935fbfccd20fa62 /bin/bash
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -p 5678:5678     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -p 4000:4000     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo
docker exec -it 947935d8f355
docker exec -it 947935d8f355 /bin/bash
docker exec -it c088188ed241 /bin/bash
docker ps
docker run -it     -v /tmp/.X11-unix:/tmp/.X11-unix     -d --gpus all     -p 8044:22     -e DISPLAY=$DISPLAY     -v /etc/timezone:/etc/timezone     -v /home:/home     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
docker ps
docker exec -it c088188ed241 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it c088188ed241
docker exec -it c088188ed241 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
docker ps -s
docker inspect 947935d8f355
docker ps 
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it c088188ed241 /bin/bash
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355
docker exec -it 947935d8f355 /bin/bash
docker exec -it 947935d8f355 /bin.bash
docker exec -it 947935d8f355 /bin/bash
docker exec -it 947935d8f355 /bin/bash
docker exec -it 947935d8f355 /bin/bash
clear
docker exec -it 947935d8f355 /bin/bash
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it c088188ed241 /bin/bash
docker ps
docker exec -it c088188ed241
docker exec -it c088188ed241 /bin/bash
docker ps
docker exec -it c088188ed241 /bin/bash
nvidia-smi
clear
docker exec -it 947935d8f355 /bin/bash
docker ps
docker exec -it 947935d8f355 /bin/bash
scp ./cxc_mmdino/AlgoServerScript cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r ./cxc_mmdino/AlgoServerScript cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/configs cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/data/smoke_fire cxc@192.168.2.46:/home/cxc/mm-grounddino/data
scp -r /media/40T/cxc_mmdino/demo cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/mmcv cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/mmdet cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/mmdet.egg-info cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/official cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/official_work_dir cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/outputs cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/projects cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/requirements cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/reaources cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/resources cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/smokefirecf_work_dir cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/tests cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/tools cxc@192.168.2.46:/home/cxc/mm-grounddino
scp -r /media/40T/cxc_mmdino/~ cxc@192.168.2.46:/home/cxc/mm-grounddino
scp  /media/40T/cxc_mmdino cxc@192.168.2.46:/home/cxc/mm-grounddino
scp  /media/40T/cxc_mmdino/* cxc@192.168.2.46:/home/cxc/mm-grounddino
nvidia-smi
docker ps
docker exec -it 947935d8f355
/bin/bash
ssh -v
ssh -V
cd ~/.ssh
cat id-rsa.pub >> authorized_keys
ll
cat id_rsa.pub >> authorized_keys
ls
ssh-keygen
cat id_rsa.pub >> authorized_keys
ls
conda env list
docker images
docker run -t -d -v /home/cxc:/home/cxc 192.168.100.25/algolib/ubuntu-anaconda:20.04 bash
docker run -it     -d --gpus all     -p 8044:22     -e DISPLAY=$DISPLAY     -v /home/cxc:/home/cxc     -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility     --shm-size 32G     --gpus all     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     192.168.100.25/algolib/mm-grounding-dino:demo /bin/bash
htop
kill -9 3012990
kill -9 3023135
kill -9 3026470
nvidia-smi
htop
kill -9 3027517
kill -9 3082084
kill -9 3041176
kill -9 2382084
htop
nvidia-smi
fuser -v /dev/nvidia*
unzip
scp -r /media/40T/cxc_mmdino/data/objects365v2/OpenDataLab___Objects365/raw/annotations/Objects365/data 192.168.2.46@cxc:/home/cxc/mm-grounddino/data/obj365
scp -r /media/40T/cxc_mmdino/data/objects365v2/OpenDataLab___Objects365/raw/annotations/Objects365/data cxc@192.168.2.46:/home/cxc/mm-grounddino/data/obj365
docker images
ls -a
rm -r .vscoder-server
rm -r ./.vscoder-server
cd ..
ls -a
cd ./cxc
ls -a
rm -r .vscode-server
ls -a
wget https://update.code.visualstudio.com/commit:695af097c7bd098fbf017ce3ac85e09bbc5dda06/server-linux-x64/stable -O vscode-server.tar.gz
ls -a
cd ~/.vscode-server
mkdir -p ~/.vscode-server/bin
cd ~/.vscode-server/bin
tar -zxf /home/cxc/vscode-server.tar.gz
mv vscode-server-linux-x64 695af097c7bd098fbf017ce3ac85e09bbc5dda06
docker ps
scp vscode-server.tar.gz cxc@10.10.8.195:/home/cxc/
ssh cxc@10.10.8.195
cat /etc/resolv.conf
nano /etc/resolv.conf
nano ~/.bashrc
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            passwd cxc
python3
docker images
docker import /home/cxc/cxcbkportsneed.tar - 192.168.100.25/algolib/mm-grounding-dino
docker ps
docker import /home/cxc/cxcbkportsneed.tar - elastic_davinci
docker import /home/cxc/cxcbkportsneed.tar elastic_davinci
docker ps
docker run -d --name cxcbkports sha256:55003292328d9cfd3cc57b919de69c8977bb4b8e07aca55d88d9fd44cef3c4a7
docker run -d --name my_container sha256:55003292328d9cfd3cc57b919de69c8977bb4b8e07aca55d88d9fd44cef3c4a7
docker run -it elastic_davinci /bin/bash
docker import /home/cxc/cxcbkportsneed.tar cxc_backports
docker run -d --name cxc_backports_cont -p 8059:22 cxc_backports 
docker run -d --name cxc_backports_cont -p 8059:22 cxc_backports /bin/bash
docker start cxc_backports_cont
docker ps -a
docker run -d --name cxc_backports_cont -p 8059:22 cxc_backports tail -f /dev/null
docker exec -it 4213944eb885 /bin/bash
docker start 4213944eb885
docker exec -it 4213944eb885 /bin/bash
docker run -d --name cxc_backports_cont -p 8059:22 cxc_backports sleep infinity
docker stop cxc_backports_cont
docker rm cxc_backports_cont
docker run -d --name cxc_backports_cont -p 8059:22 cxc_backports sleep infinity
docker stop cxc_backports_cont
docker rm cxc_backports_cont
docker run -it --gpus all -e DISPLAY=$DISPLAY -v /etc/timezone:/etc/timezone -v /home:/home -v /home/cxc:/home/cxc -e NVIDIA_DRIVER_CAPABILIYIES=vedio,compute,utility --shm-size 32G --gpus all --cap-add=SYS_PTRACE seccomp=unconfined --name cxc_backports_cont -p 8059:22 cxc_backports sleep infinity
docker run -d --gpus all -e DISPLAY=$DISPLAY -v /etc/timezone:/etc/timezone -v /home:/home -v /home/cxc:/home/cxc -e NVIDIA_DRIVER_CAPABILIYIES=vedio,compute,utility --shm-size 32G --gpus all --cap-add=SYS_PTRACE seccomp=unconfined --name cxc_backports_cont -p 8059:22 cxc_backports sleep infinity
docker run -d --gpus all -e DISPLAY=$DISPLAY -v /etc/timezone:/etc/timezone -v /home:/home -v /home/cxc:/home/cxc -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --shm-size 32G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name cxc_backports_cont -p 8059:22 cxc_backports sleep infinity
docker -it exec 21638a6761de
docker  exec -it  21638a6761de /bin/bash
docker stop cxc_backports_cont
docker rm cxc_backports_cont
docker run -d --gpus all -e DISPLAY=$DISPLAY -v /etc/timezone:/etc/timezone -v /home:/home -v /home/cxc:/home/cxc -v /usr/local/python3.9.18:/usr/local/python3.9.18 -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --shm-size 32G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name cxc_backports_cont -p 8059:22 cxc_backports sleep infinity
docker stop cxc_backports_cont
docker rm cxc_backports_cont
