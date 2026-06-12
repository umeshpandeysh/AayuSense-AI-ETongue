# Rasa Profiling — Ayurvedic Taste Classification

## Overview

One of AayuSense's unique contributions is **Rasa profiling** — an experimental approach
that maps electrochemical sensor readings from the ESP32 array to the six taste categories
(Rasas) used in Ayurvedic herbal quality assessment.

In Ayurveda, each herb has a characteristic Rasa (taste) profile that indicates its
therapeutic properties. Adulteration typically disrupts this natural profile.

## The Six Rasas

| Rasa | Devanagari | English | Therapeutic Role |
|------|-----------|---------|------------------|
| Madhura | मधुर | Sweet | Nourishing, cooling, building |
| Amla | अम्ल | Sour | Digestive, warming |
| Lavana | लवण | Salty | Softening, hydrating |
| Tikta | तिक्त | Bitter | Detoxifying, anti-inflammatory |
| Katu | कटु | Pungent | Stimulating, metabolic |
| Kashaya | कषाय | Astringent | Binding, healing |

## Sensor → Rasa Mapping

The mapping is based on the electrochemical properties associated with each taste:

### Madhura (Sweet)
- **Primary sensors**: pH (+), TDS (+)
- **Inverse sensors**: ORP (−)
- **Rationale**: Sweet compounds are generally neutral to slightly alkaline with moderate dissolved solids and low oxidation potential.

### Amla (Sour)
- **Primary sensors**: pH (−, i.e. acidic), ORP (+)
- **Rationale**: Acidic pH is the direct electrochemical correlate of sour taste. High ORP confirms oxidative character.

### Lavana (Salty)
- **Primary sensors**: TDS (+), Salt_content (+), Ionic_value (+)
- **Rationale**: Salinity directly increases TDS, ionic concentration, and salt content readings.

### Tikta (Bitter)
- **Primary sensors**: ORP (+), Reduction_value (+)
- **Inverse sensors**: Turbidity (−)
- **Rationale**: Bitter alkaloids have high redox activity. Clear solutions (low turbidity) are typical.

### Katu (Pungent)
- **Primary sensors**: ORP (+), Reduction_value (+), Temperature (+)
- **Rationale**: Pungent compounds (e.g., capsaicin, piperine) have high reduction potential and are thermally active.

### Kashaya (Astringent)
- **Primary sensors**: Ionic_value (+), Reduction_value (+), Turbidity (+)
- **Rationale**: Tannins (astringent agents) bind ions (high ionic), are oxidatively active, and increase turbidity.

## Implementation

Rasa intensities are computed in `src/feature_engineering.py`:

```python
from src.feature_engineering import compute_rasa_intensities
df_with_rasa = compute_rasa_intensities(df)
# Adds columns: rasa_madhura, rasa_amla, rasa_lavana, rasa_tikta, rasa_katu, rasa_kashaya
```

Each Rasa intensity is a weighted linear combination of normalized sensor values,
clipped to [0, 1].

## Important Note

The Rasa mapping in this implementation is **experimental** — it represents the
AayuSense team's domain-knowledge hypothesis about sensor-Rasa correlations.
Validation against certified Ayurvedic reference samples would be required
to confirm the mapping quantitatively.

## Example Output

For Neem (Azadirachta indica) — a predominantly Tikta (bitter) herb:

```json
{
  "rasa_madhura": 0.22,
  "rasa_amla":    0.41,
  "rasa_lavana":  0.18,
  "rasa_tikta":   0.87,
  "rasa_katu":    0.65,
  "rasa_kashaya": 0.73
}
```

This matches the expected Ayurvedic profile of Neem: predominantly Tikta (bitter)
with Kashaya (astringent) secondary character.
