"""
Start-up patch for the BioChatter Light container (biocypher/biochatter-light:0.9.0).

The app only uses a Google API key for models in BioChatter's hard-coded
GEMINI_MODELS list (gemini-2.0-flash, gemini-2.5-flash-preview-04-17), which are
no longer available to new keys. Register the configured model
(BIOCHATTER_DEFAULT_MODEL) so the key is passed to it. Python imports this module
automatically at start-up because its directory is on PYTHONPATH.
"""

import os

model = os.getenv("BIOCHATTER_DEFAULT_MODEL")
if model and model.startswith("gemini"):
    try:
        from biochatter.llm_connect import available_models

        if model not in available_models.GEMINI_MODELS:
            available_models.GEMINI_MODELS.append(model)
        available_models.TOKEN_LIMITS.setdefault(model, 1_048_576)
    except Exception as e:  # never block the app from starting
        print(f"sitecustomize: could not register {model}: {e}")
