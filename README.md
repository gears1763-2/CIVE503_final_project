# Surrogate Modelling Efficiency

    Anthony Truelove MASc, P.Eng.
    Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
SEE LICENSE TERMS [HERE](./LICENSE)

This is a final project completed in partial fulfillment of the course
requirements of CIVE 503, as offered at the University of Victoria by
Dr. Ralph Evins (winter 2025).

--------


## Contents

In the directory for this project, you should find this README, a LICENSE file,
a bash script for generating the Python documentation, and the following
sub-directories:

    data/           For holding data sets generated in the course of this
                    project.

    presentation/   For holding presentations given in regards to this work.

    python/         For holding the supporting Python scripts (including tests
                    and documentation).

    tex/            For holding the main paper (including source and assets).

--------


## Autodocumentation

Autodocumentation for this project is achieved using `pdoc` (see
<https://pdoc.dev/>). That said, before making use of the `make_docs`
script, you need to make sure that `pdoc` is installed. Fortunately, this is 
easily done using `pip`:

    pip install pdoc

Finally, note the autodocumentation includes generating a `pip` requirements
file within the `python/` directory. This depends on the `pipreqs` package, so
be sure to install that as well before making use of the `make_docs`
script. Again, it's `pip` installable:

    pip install pipreqs

--------
