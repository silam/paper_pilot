.PHONY: install ingest api ui test eval compare demo up down k8s lint

install:        ## local dev install (no GPU / no keys needed)
	pip install -r requirements-dev.txt

ingest:         ## build the index from the bundled sample papers
	python -m app.ingestion.pipeline --source sample

api:            ## run the API with hot reload -> http://localhost:8000/docs
	uvicorn app.main:app --reload

ui:             ## Streamlit UI -> http://localhost:8501
	streamlit run ui/streamlit_app.py

demo:           ## scripted end-to-end walkthrough in the terminal
	python scripts/demo.py

test:
	pytest -q

lint:
	ruff check app eval tests

eval:           ## offline evaluation + quality gate
	python -m eval.run_eval --gate

compare:        ## offline prompt A/B
	python -m eval.run_eval --compare rag_v1 rag_v2_fewshot

up:             ## full stack: API, worker, UI, Postgres, Redis, Chroma, Prometheus, Grafana
	docker compose up --build -d

down:
	docker compose down

k8s:
	kubectl apply -f k8s/
