import numpy as np
import matplotlib.pyplot as plt

emb = np.load("./prototyping/encoded/speaker_embedding.npy")
print("Shape:", emb.shape)       # (1, 192)
print("Type:", type(emb))
print("Min/Max:", emb.min(), emb.max())
print("Norm:", np.linalg.norm(emb))

# Make it 1D
emb_1d = emb.squeeze()  # removes dimensions of size 1

plt.figure(figsize=(10, 3))
plt.bar(range(len(emb_1d)), emb_1d)
plt.title("Speaker Embedding Vector")
plt.show()
