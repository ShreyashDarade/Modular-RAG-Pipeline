"""Ports: the small interfaces the orchestrators depend on (dependency inversion).

Each protocol is deliberately narrow (interface segregation) so an implementation only has to
provide what its consumers call: writers are separate from searchers, embedders from chat
models, parsers from OCR engines.
"""
