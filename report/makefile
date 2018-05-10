PNGS = $(find ./pictures -type f -name "*.png")
JPGS = $(find ./pictures -type f -name "*.jpg")
TEXFILES = $(shell ls *.tex)

.PHONY: clean

all: report.pdf

report.blg: literature.bib
	pdflatex report.tex -halt-on-error -draftmode
	bibtex report.aux

report.pdf: report.blg makefile $(TEXFILES) $(PNGS) $(JPGS)
	pdflatex report.tex -halt-on-error -draftmode
	pdflatex report.tex -halt-on-error

clean:
	rm -f report.aux report.bbl report.blg report.log report.out report.pdf report.toc
