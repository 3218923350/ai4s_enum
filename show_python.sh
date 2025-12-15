for ip in 47.242.24.80 47.242.162.197 8.217.119.67 47.238.72.118 47.238.100.201 8.217.84.168 47.238.112.245 47.238.99.156 8.210.254.104
do
  echo "====== $ip ======"
  ssh root@$ip "python -m venv .enum && source .enum/bin/activate && python -m pip install -r requirements.txt && nohup python run.py --all > log 2>&1 &" 
done
