install:
	pip3 install -r requirements.txt

run:
	python3 -m gunicorn main:app \
		--workers 1 \
		--worker-class gthread \
		--bind :5000 \
		--timeout 0

build_docker:
	docker image build -t text_moderation .

run_docker:
	docker run --env PORT=5000 -p 5000:5000 -d text_moderation

venv:
	python3 -m venv env
	source env/bin/activate