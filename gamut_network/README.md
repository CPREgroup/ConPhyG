# Physically-guided Image Generation for Multi-Projection Mapping

`train_gamut_distance_network.py` in `gamut_network` generates a synthetic gamut dataset for the 4-projector setup and trains the `gamut_distance_network_39d.pth` model.

**⚠️ IMPORTANT**: You MUST have an active **Gurobi license** to run this code, otherwise the data generation will fail.
