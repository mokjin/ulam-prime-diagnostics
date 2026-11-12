# Ulam prime discovery diagnostics

Branden Kim, Virginia Tech

**First-Discovery Distances and Sieve-Based Model Diagnostics on Ulam Spiral Paths**

Finite computational experiments on first-prime discovery along Ulam diagonal paths, with sieve-based independent survival models and fixed-support residual diagnostics. This is a computational case study, not a new asymptotic prime law or a faster primality algorithm.

## Complete reproducibility package

Download **ulam-prime-diagnostics.zip** from the [v0.1.0 release](https://github.com/mokjin/ulam-prime-diagnostics/releases/tag/v0.1.0). It contains the original directory structure, generated data, all scripts, frozen protocol and coefficients, checksums, manuscript Word/PDF, figures and reproduction instructions. The source files here permit browsing; use the release ZIP for the complete runnable snapshot. GitHub's automatically generated source-code ZIP is not the full data package.

After extracting the release asset:

```sh
cd ulam-prime-diagnostics
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/run_all.py
```

The packaged verifier checks archived file hashes, 17 unit tests, all four frozen-model survival errors and the fixed-support baseline contrast. Full replay runs separately in a new directory and never overwrites archived results. The package README documents runtime requirements and limitations.

## Results and chronology

Small-prime exclusions improve survival prediction over unfiltered density. Increasing the sieve cutoff does not consistently improve aggregate errors. Pair-score residual departures in a frozen evaluation substantially diminish after additional candidate-wise exclusions on fixed support. That last analysis is explicitly post-evaluation, not a fresh frozen replication. Conditional simulations do not generate the prime-anchor selection process and are not a test of prime independence.

## License

Code and software documentation: MIT. Generated data and figures: CC BY 4.0. The manuscript is excluded from both grants. See LICENSE and LICENSE-DATA.md.

## AI assistance and responsibility

ChatGPT and Codex assisted research planning, code development, literature discovery and writing. Codex invoked execution tools that ran Python scripts and tests. Branden Kim reviewed the manuscript and is responsible for the work. The manuscript gives the full disclosure.

## Archive status

GitHub release publication and archival DOI registration are separate. No Zenodo DOI has been issued for this version.
