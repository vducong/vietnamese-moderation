install:
	pip3 install -r requirements.txt

run:
	export FLASK_APP=main.py && python3 -m flask run