sudo -i
exit
echo "conda activate mindspore" >> ~/.bashrc
echo "export GLOG_v=2" >> ~/.bashrc
echo "LOCAL_ASCEND=/usr/local/Ascend" >> ~/.bashrc
echo "source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh" >> ~/.bashrc