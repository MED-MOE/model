# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
pip install --upgrade pip

python3 -m virtualenv -p python3.9 helm-venv

# Activate the virtual environment.
source helm-venv/bin/activate

sudo apt-get update
sudo apt-get install python3.9-dev
pip install blis
pip install --upgrade pip
pip install crfm-helm


# Download configurations to your root folder
https://github.com/stanford-crfm/helm/blob/8c2fa4b6bab791c1dc3285ec3fdd63427f92b837/src/helm/benchmark/presentation/run_entries_biomedical.conf
