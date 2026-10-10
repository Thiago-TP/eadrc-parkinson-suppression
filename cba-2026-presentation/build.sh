# This is a simple build script for the CBA 2026 presentation.
# It runs pdflatex and biber in the correct order to generate the final PDF.
# To run it, simply execute this script in the terminal with `./build.sh`.
# You might need to give it execute permissions first with `chmod +x build.sh`.
# In Windows, you can run it in Git Bash or any other bash shell.
# The slides reuse figures and references from the paper, in ../cba-2026-paper.

# Run from the script's folder, so that it also works when called from elsewhere
cd "$(dirname "$0")"

# Name of the main tex file to be compiled (minus the .tex extension)
FILE=presentation

# Build commands (loud version)
# pdflatex $FILE # first pass that builds most of the document
# biber $FILE # second pass that builds the bibliography
# pdflatex $FILE # third pass that builds the document with the bibliography
# pdflatex $FILE # fourth pass that resolves citations, cross-references and frame counts

# Build commands (quiet version)
pdflatex -interaction=batchmode $FILE # first pass that builds most of the document
biber --quiet $FILE # second pass that builds the bibliography
# Further passes build the document with the bibliography, then resolve citations,
# cross-references and frame counts, repeating while LaTeX asks for a rerun (up to 4 times)
for PASS in 1 2 3 4; do
    pdflatex -interaction=batchmode $FILE
    grep -qE "Rerun to get|Please rerun" $FILE.log || break
done

# Report errors, warnings and content overflowing the slides left after the last pass
# (no output means a clean build)
grep -E "^!|Warning|WARN|ERROR|Overfull" $FILE.log $FILE.blg

# Clean up auxiliary files generated during the build process (optional)
rm -f $FILE.aux $FILE.bbl  $FILE.bcf $FILE.blg $FILE.log $FILE.nav $FILE.out $FILE.run.xml $FILE.snm $FILE.synctex.gz $FILE.toc
