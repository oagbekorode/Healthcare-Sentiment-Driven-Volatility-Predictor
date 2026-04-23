"""
Biotech ticker universe: keep prices, news search queries, and docs in sync.
GNews `q` uses the company name (broader match than raw ticker symbols).
"""

BIOTECH_UNIVERSE: dict[str, str] = {
    "MRNA": "Moderna",
    "VRTX": "Vertex Pharmaceuticals",
    "BNTX": "BioNTech",
    "REGN": "Regeneron Pharmaceuticals",
    "BIIB": "Biogen",
    "ALNY": "Alnylam Pharmaceuticals",
    "INCY": "Incyte",
    "BMRN": "BioMarin Pharmaceutical",
    "SGEN": "Seagen",
    "HZNP": "Horizon Therapeutics",
}


def tickers() -> list[str]:
    return list(BIOTECH_UNIVERSE.keys())
