# This is a simple build script for the CBA 2026 paper.
# It runs pdflatex and bibtex in the correct order to generate the final PDF.
# To run it, simply execute this script in the terminal with `./build.sh`.
# You might need to give it execute permissions first with `chmod +x build.sh`.
# In Windows, you can run it in Git Bash or any other bash shell.

# Run from the script's folder, so that it also works when called from elsewhere
cd "$(dirname "$0")"

# Name of the main tex file to be compiled (minus the .tex extension)
FILE=paper

# Build commands (loud version)
# pdflatex $FILE # first pass that builds most of the document
# bibtex $FILE # second pass that builds the bibliography
# pdflatex $FILE # third pass that builds the document with the bibliography
# pdflatex $FILE # fourth pass that resolves citations and cross-references

# Build commands (quiet version)
pdflatex -interaction=batchmode $FILE # first pass that builds most of the document
bibtex -terse $FILE # second pass that builds the bibliography (bibtex has no -interaction option)
# Further passes build the document with the bibliography, then resolve citations and
# cross-references, repeating while LaTeX asks for a rerun (up to 4 times)
for PASS in 1 2 3 4; do
    pdflatex -interaction=batchmode $FILE
    grep -qE "Rerun to get|Please rerun" $FILE.log || break
done

# Report errors and warnings left after the last pass (no output means a clean build)
grep -E "^!|Warning" $FILE.log $FILE.blg

# Clean up auxiliary files generated during the build process (optional)
rm -f $FILE.aux $FILE.bbl $FILE.blg $FILE.log
