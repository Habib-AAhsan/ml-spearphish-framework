.PHONY: demo run test

demo:
	python3 -m venv .venv || true
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && python scripts/toy_data.py --n 3 --out data/toy.json &
	. .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8000 &

run:
	. .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8000

test:
	python scripts/load_demo.py data/toy.json
