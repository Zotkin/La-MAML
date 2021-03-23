apt-get install -y vim git
mkdir /code && mkdir /code/data && mkdir /code/logs
cd /code
git clone https://github.com/Zotkin/La-MAML.git
pip install -r requirements.txt