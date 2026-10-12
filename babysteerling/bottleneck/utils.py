def _out_concepts_and_n(out_concepts):
    if isinstance(out_concepts, int):
        out_concepts = [f"concept_{i}" for i in range(out_concepts)]
    return out_concepts, len(out_concepts)
