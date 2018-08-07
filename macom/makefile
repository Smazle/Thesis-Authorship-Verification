TEXFILES = $(shell ls *.tex)

.PHONY: clean

all: summary.pdf

summary.blg: literature.bib
	pdflatex summary.tex -halt-on-error -draftmode
	bibtex summary.aux

summary.pdf: summary.blg makefile $(TEXFILES)
	pdflatex summary.tex -halt-on-error -draftmode
	pdflatex summary.tex -halt-on-error
	! pdftotext summary.pdf - | grep -n --color " ? \|??\|\.?"

clean:
	rm -f summary.aux summary.bbl summary.blg summary.log summary.out summary.pdf summary.toc
