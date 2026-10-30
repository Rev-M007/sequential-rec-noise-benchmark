"""Custom RecBole sequential models for this benchmark."""

from models.fmlp_rec import FMLPRec
from models.sig4rec import Sig4Rec, prepare_ml1m_genres

__all__ = ["FMLPRec", "Sig4Rec", "prepare_ml1m_genres"]
