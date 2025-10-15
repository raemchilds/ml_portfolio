import sys, os
# make sure we can import your converted module from the current folder
sys.path.append(os.getcwd())

import streamlit as st
from Glowlock_Sensory_Engine import GSEngine, SAMPLE_REALMS

st.set_page_config(page_title="Glowlock Sensory Engine", page_icon="🌲", layout="wide")
st.title("🌲 Glowlock Sensory Engine")
st.caption("Narrative worldbuilding meets machine learning · Glowlock Labs © 2025")

@st.cache_resource
def load_engine():
    gse = GSEngine(SAMPLE_REALMS)
    gse.fit_embeddings()
    return gse

# 🌸 Friendly loading UI
with st.status("🌸 Setting up Glowlock Sensory Engine…", expanded=True) as status:
    st.write("• Downloading the embedding model if needed (`all-MiniLM-L6-v2`).")
    st.write("• Computing realm embeddings (first run is the longest).")
    gse = load_engine()
    status.update(label="✨ Ready! Type a vibe in the sidebar.", state="complete", expanded=False)
st.toast("Glowlock is ready ✨", icon="🌟")


# — Sidebar —
st.sidebar.header("🔍 Vibe Search")
query = st.sidebar.text_input(
    "Describe a feeling, scene, or vibe:",
    "dreamy forest rain with honey and flute sounds"
)
top_k = st.sidebar.slider("Number of results", 1, 5, 3)
show_map = st.sidebar.checkbox("Show 2D vibe map")

# — Main —
if st.sidebar.button("Search"):
    results = gse.generate_prompt(query, top_k=top_k)
    for name, score, prompt in results:
        st.subheader(f"{name} · similarity {score:.3f}")
        st.code(prompt, language="markdown")

if show_map:
    import matplotlib.pyplot as plt
    coords = gse.umap_2d()
    xs, ys = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    for name, x, y in zip(gse._names, xs, ys):
        ax.text(x, y, name, fontsize=9, ha="center")
    ax.set_title("Glowlock Realm Vibe Map (UMAP)")
    st.pyplot(fig)
