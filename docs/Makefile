# Build the CellFlow manuals.
#   make            # build both PDFs
#   make clean      # remove build artifacts (including PDFs)
LATEX := pdflatex -interaction=nonstopmode -halt-on-error
DOCS  := technical_manual user_manual

.PHONY: all clean
all: $(addsuffix .pdf,$(DOCS))

%.pdf: %.tex
	$(LATEX) $< >/dev/null
	$(LATEX) $< >/dev/null   # second pass: table of contents + cross-references
	@echo "built $@"

clean:
	rm -f *.aux *.log *.out *.toc *.pdf
