from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import json

# ----------------------------
# Sample sensory profiles
# ----------------------------
SAMPLE_REALMS: Dict[str, Dict[str, str]] = {
    "Moss Merry Way": {
        "taste": "sweet rain, mint, forest honey",
        "feel": "spongy moss, gentle drizzle",
        "color": "emerald, misty teal",
        "sound": "rustling leaves, flute tones",
        "smell": "wet earth, pine sap"
    },
    "Jingle Hooves": {
        "taste": "vanilla snow, peppermint",
        "feel": "crisp winter air, velvet mittens",
        "color": "frost white, crimson",
        "sound": "sleigh bells, laughter",
        "smell": "cinnamon, cold metal"
    },
    "Dusk Hallows": {
        "taste": "smoked berries, ink",
        "feel": "velvet drapes, electric tension",
        "color": "violet, black ink",
        "sound": "whispered lullabies, thunder hum",
        "smell": "candles, wet stone"
    },
    "Pearl Mist": {
        "taste": "salted caramel foam, citrus spray",
        "feel": "cool lagoon ripple, glassy stillness",
        "color": "opal, pale gold flecks",
        "sound": "harp glissando, distant gulls",
        "smell": "sea lavender, sun-warmed shells"
    },
    "Raspberry Ranch": {
        "taste": "ripe raspberry jam, shortbread",
        "feel": "sun-warmed straw, twine",
        "color": "rose, coral red",
        "sound": "bee hum, wooden wind chimes",
        "smell": "berry bramble, fresh cream"
    }
}

# ----------------------------
# Helpers
# ----------------------------
def profile_to_text(profile: Dict[str, str]) -> str:
    parts = []
    for key in ("taste", "feel", "color", "sound", "smell"):
        if key in profile and profile[key]:
            parts.append(f"{key}: {profile[key]}")
    return " | ".join(parts)

def cosine_sim_matrix(A, B):
    import numpy as np
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T

def _split_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

# ----------------------------
# Core Engine
# ----------------------------
@dataclass
class GSEngine:
    realms: Dict[str, Dict[str, str]]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: Any = field(default=None, init=False, repr=False)
    _names: List[str] = field(default_factory=list, init=False)
    _docs: List[str] = field(default_factory=list, init=False)
    _X: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._names = list(self.realms.keys())
        self._docs = [profile_to_text(self.realms[name]) for name in self._names]

    # Embeddings
    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit_embeddings(self) -> "GSEngine":
        self._ensure_model()
        self._X = self._model.encode(self._docs, convert_to_numpy=True, normalize_embeddings=False)
        return self

    # Search
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._X is None:
            raise RuntimeError("Embeddings not fit. Call .fit_embeddings() first.")
        self._ensure_model()
        q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        S = cosine_sim_matrix(q, self._X)[0]
        idxs = S.argsort()[::-1][:top_k]
        return [(self._names[i], float(S[i])) for i in idxs]

    # UMAP + plot
    def umap_2d(self, n_neighbors: int = 8, min_dist: float = 0.25, random_state: int = 42):
        if self._X is None:
            raise RuntimeError("Embeddings not fit. Call .fit_embeddings() first.")
        import umap
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)\
                   .fit_transform(self._X)

    def plot_vibe_map(self, annotate: bool = True):
        coords = self.umap_2d()
        import matplotlib.pyplot as plt
        xs, ys = coords[:, 0], coords[:, 1]
        plt.figure()
        plt.scatter(xs, ys)
        if annotate:
            for name, x, y in zip(self._names, xs, ys):
                plt.annotate(name, (x, y), xytext=(5, 3), textcoords="offset points")
        plt.title("Glowlock Sensory Engine — Vibe Map (UMAP)")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout(); plt.show()

    # Prompting
    def build_prompt_from_profile(self, name: str, profile: Dict[str, str]) -> str:
        taste = ", ".join(_split_list(profile.get("taste", "")))
        feel  = ", ".join(_split_list(profile.get("feel", "")))
        color = ", ".join(_split_list(profile.get("color", "")))
        sound = ", ".join(_split_list(profile.get("sound", "")))
        smell = ", ".join(_split_list(profile.get("smell", "")))
        return (
            f"{name} — storybook cinematic, 2D animation\n"
            f"palette: {color} | atmosphere: {feel} | sounds: {sound}\n"
            f"taste/scent: {taste}, {smell}\n"
            f"Glowlock Labs aesthetic, whimsical, cozy, magical realism"
        )

    def generate_prompt(self, query: str, top_k: int = 1) -> List[Tuple[str, float, str]]:
        hits = self.search(query, top_k=top_k)
        out: List[Tuple[str, float, str]] = []
        for name, score in hits:
            out.append((name, score, self.build_prompt_from_profile(name, self.realms[name])))
        return out

# Keep all demo code inside this guard so imports are clean
if __name__ == "__main__":
    print("Glowlock Sensory Engine — CLI Demo")
    gse = GSEngine(SAMPLE_REALMS).fit_embeddings()
    print(gse.search("dreamy forest rain with honey and flute sounds", top_k=3))




