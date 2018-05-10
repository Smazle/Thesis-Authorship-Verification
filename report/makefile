PICTURES = $(find ./pictures -type f -name "*.png")
TEXFILES = $(shell ls *.tex)

.PHONY: clean

all: report.pdf

report.blg: literature.bib
	pdflatex report.tex
	bibtex report.aux

report.pdf: report.blg makefile $(PICTURES) $(TEXFILES)
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f report.aux report.bbl report.blg report.log report.out report.pdf report.toc
