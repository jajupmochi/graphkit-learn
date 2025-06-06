sphinx-apidoc -o docs/ gklearn/ --separate

sphinx-apidoc -o source/ ../gklearn/ --separate --force --module-first --no-toc

make html
