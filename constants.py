"""
constants.py — Única fuente de verdad para grupos de discurso
==============================================================

Define los identificadores de grupo canónicos y proporciona
compatibilidad hacia atrás con artefactos que usen "political".
"""

# ── Grupos canónicos ────────────────────────────────────────────────
GROUPS = {"academic", "institutional", "public"}

# ── Compatibilidad con artefactos legacy ────────────────────────────
LEGACY_MAP = {"political": "institutional"}


def normalize_group(x: str) -> str:
    """Mapea nombres legacy a los canónicos.

    Parameters
    ----------
    x : str
        Nombre de grupo, puede ser canónico o legacy.

    Returns
    -------
    str
        Nombre canónico del grupo.

    Examples
    --------
    >>> normalize_group("political")
    'institutional'
    >>> normalize_group("academic")
    'academic'
    """
    if not isinstance(x, str):
        return x
    return LEGACY_MAP.get(x.lower().strip(), x.lower().strip())


def normalize_group_series(series):
    """Aplica normalize_group a una Serie de pandas.

    Parameters
    ----------
    series : pd.Series
        Serie con nombres de grupo (puede contener legacy).

    Returns
    -------
    pd.Series
        Serie con nombres canónicos.
    """
    return series.map(normalize_group)
