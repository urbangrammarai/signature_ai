build-book:
	rm -rf docs
	rm -rf book/_build
	# list folders with notebooks here. Notebooks must be present in _toc.yml.
	cp -r ai_experiments book/ai_experiments
	cp -r create_chips book/create_chips
	jupyter-book build book
	cp -r book/_build/html docs
	rm -rf book/ai_experiments
	rm -rf book/create_chips
	touch docs/.nojekyll