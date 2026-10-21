"""demo birds and a generated example clip."""

from functools import lru_cache
from io import BytesIO

import numpy as np
import soundfile as sf

# placeholder birds for the demo, not the final model classes
# facts are based on the linked DOC pages
SPECIES = [
    {"label": "morepork", "display_name": "Morepork", "maori_name": "Ruru",
     "fact": "Ruru are often heard calling in forests at dusk and through the night.",
     "source": "https://www.doc.govt.nz/nature/native-animals/birds/birds-a-z/morepork-ruru/"},
    {"label": "bellbird", "display_name": "Bellbird", "maori_name": "Korimako",
     "fact": "Korimako are melodious songbirds found across many New Zealand islands.",
     "source": "https://www.doc.govt.nz/bellbird"},
    {"label": "tui", "display_name": "Tūī", "maori_name": "Tūī",
     "fact": "Tūī can mimic other birds and even sounds such as phone ringtones.",
     "source": "https://www.doc.govt.nz/nature/native-animals/birds/birds-a-z/tui/"},
    {"label": "fantail", "display_name": "Fantail", "maori_name": "Pīwakawaka",
     "fact": "Pīwakawaka are known for energetic flight and their distinctive cheet-cheet call.",
     "source": "https://www.doc.govt.nz/nature/native-animals/birds/birds-a-z/fantail-piwakawaka/"},
    {"label": "kiwi", "display_name": "Kiwi", "maori_name": "Kiwi",
     "fact": "New Zealand has five species of kiwi; this demo groups them under one label.",
     "source": "https://www.doc.govt.nz/nature/native-animals/birds/birds-a-z/kiwi/facts/"},
    {"label": "kakapo", "display_name": "Kākāpō", "maori_name": "Kākāpō",
     "fact": "Kākāpō are nocturnal, flightless parrots with an owl-like face.",
     "source": "https://www.doc.govt.nz/nature/native-animals/birds/birds-a-z/kakapo/"},
    {"label": "kea", "display_name": "Kea", "maori_name": "Kea",
     "fact": "Kea are New Zealand parrots renowned for their problem-solving abilities.",
     "source": "https://www.doc.govt.nz/nature/native-animals/birds/birds-a-z/kea/"},
    {"label": "kokako", "display_name": "Kōkako", "maori_name": "Kōkako",
     "fact": "Kōkako belong to New Zealand's wattlebird family, alongside saddlebacks.",
     "source": "https://www.doc.govt.nz/kokako"},
]


@lru_cache(maxsize=1)
def example_clip() -> bytes:
    """make a five-second tone without saving an audio file."""
    sample_rate = 32000
    t = np.arange(5 * sample_rate, dtype=np.float64) / sample_rate
    frequency = 1100 + 650 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    envelope = np.sin(np.pi * t / 5) ** 2 * (0.2 + 0.8 * np.sin(2 * np.pi * 1.5 * t) ** 4)
    y = (0.32 * envelope * (np.sin(phase) + 0.25 * np.sin(2 * phase))).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, y, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
