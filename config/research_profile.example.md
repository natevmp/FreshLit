# FreshLit Research Profile (Example)

This is a template. Copy it to `config/research_profile.md` and edit it to
describe your own research. The **Keywords** list feeds OpenAlex/Europe PMC
queries; the **Research Description** feeds the vector-similarity profile
(`build-profile`) and the LLM relevance rubric.

After editing `config/research_profile.md`, run
`python -m freshlit.main build-profile` to refresh the vector profile.

## Keywords

- clonal haematopoiesis
- somatic evolution
- clonal dynamics
- fitness landscape
- mutational signature
- population genetics inference

## Research Description

Describe your research focus here. Be specific about the mathematical and
computational methods you use (e.g. stochastic processes, Bayesian inference,
PDEs, population genetics) and the data types you work with (e.g. sequencing,
lineage tracing, single-cell/clonal-growth data). Clearly state what is out of
scope (e.g. purely clinical or purely experimental papers).

Relevant papers typically combine mathematical or statistical modelling with
experimental or observational data.
