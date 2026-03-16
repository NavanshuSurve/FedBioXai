## Setup

1. Clone the repository

git clone <repo>

2. Create a virtual environment

python -m venv venv

3. Activate it

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

5. Run federated training

python server.py
python run_clients.py
