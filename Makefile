freeze:
	@pip freeze > requirements.txt
	@git add -A :/
	@git commit --amend --no-edit

install:
	@python3 -m venv .env
	@source .env/bin/activate
	@pip install -r requirements.txt

activate:
	@source .env/bin/activate

api:
	@FLASK_APP=./api.py FLASK_DEBUG=1 flask run

clean:
	@rm -r ./.dummy/*
