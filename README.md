# 📈 Cycle Engine — Market Cycle Analysis Terminal

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Personal%20Project-lightgrey)

Eine interaktive Streamlit-App zur Analyse dominanter Preiszyklen bei Aktien und Kryptowährungen mittels klassischer Signalverarbeitung — **kein Machine Learning, keine Blackbox**.

> ⚠️ Analyse- und Lernprojekt. **Keine Anlageberatung, keine verlässliche Kursprognose.**

---

## Inhaltsverzeichnis

- [Features](#features)
- [Wie es funktioniert](#wie-es-funktioniert)
- [Bedienbare Parameter](#bedienbare-parameter-sidebar)
- [Verwendete Technologien](#verwendete-technologien)
- [Installation](#installation)
- [Nutzung](#nutzung)
- [Hinweis](#hinweis)

---

## Features

- 🔍 Automatische Erkennung dominanter Marktzyklen (16–420 Tage) in Aktien- und Krypto-Kursen
- 📊 Klassifizierung in Long-, Medium- und Short-Term-Zyklen
- 🌀 Kombinierte "Composite Mix"-Ansicht mehrerer Zyklen
- 🔮 Projektion möglicher zukünftiger Wendepunkte auf Basis historischer Zyklusmuster
- 🎛️ Manuelle Zyklusauswahl statt automatischer Bestwahl
- 📉 Interaktive Plotly-Charts mit Tops/Bottoms-Markierung und Cycle-Spectrum-Ansicht

## Wie es funktioniert

1. **Datenabruf** — Kursdaten via `yfinance` (inoffizielle Bibliothek, kein API-Key nötig; Yahoo hat 2017 die offizielle Finance-API eingestellt)
2. **Detrending** — Logarithmierter Kurs minus gleitender Durchschnitt (Fenster 40–160 Tage)
3. **Bandpassfilterung** — Butterworth-Filter 2. Ordnung (`scipy.signal.butter` + `filtfilt`) pro getesteter Zykluslänge
4. **Zyklus-Fit** — Gewichtete Sinus/Kosinus-Regression (`numpy.linalg.lstsq`), aktuellere Daten stärker gewichtet
5. **Phasenqualität** — Hilbert-Transformation misst die Konsistenz des Zyklus-Timings
6. **Stabilität** — Signal wird in 3–5 Abschnitte geteilt; Konsistenz der Korrelation über die Abschnitte ergibt den Stabilitätswert
7. **Score-Formel** (exakt aus dem Code):
   ```
   Qualität = 0,45 × Korrelation + 0,25 × Signalstärke + 0,20 × Stabilität + 0,10 × Phasenqualität
   ```
8. **Klassifizierung** — Long ≥150 Tage · Medium ≥55 Tage · Short <55 Tage (Duplikate <18 Tage Abstand werden gefiltert)
9. **Projektion & Tops/Bottoms** — Rücktransformation in den Preisraum, Fortschreibung in die Zukunft, Peak-Erkennung via `scipy.signal.find_peaks`

## Bedienbare Parameter (Sidebar)

| Parameter | Optionen |
|---|---|
| Asset-Ticker | Freitext, z. B. `AAPL`, `BTC-USD`, `TSLA` |
| Cycle View | Long Term / Mid Term / Short Term / Composite Mix |
| Historie | 1y / 2y / 5y / max |
| In-Sample-Zeitraum | Start-/Enddatum (mind. 180 Tageskerzen erforderlich) |
| Y-Achse | Linear / Logarithmisch |
| Anzeige-Toggles | Tops, Bottoms, Projektion, Spektrum |
| Zyklusauswahl | Auto oder manuell aus Top-12-Liste |

## Verwendete Technologien

- **Python** · **Streamlit** (UI)
- **yfinance** — inoffizielle Kursdaten-Bibliothek für Yahoo Finance
- **NumPy / SciPy** — Butterworth-Bandpassfilter, Hilbert-Transformation, Peak-Erkennung, lineare Regression
- **Pandas** — Datenverarbeitung
- **Plotly** — interaktive Visualisierung

## Installation

```bash
git clone https://github.com/mekanxy-art/cycle-engine-market-analysis.git
cd cycle-engine-market-analysis
pip install -r requirements.txt
```

**requirements.txt**
```
streamlit
yfinance
pandas
numpy
scipy
plotly
```

## Nutzung

```bash
streamlit run cycle_engine.py
```

Die App öffnet sich automatisch im Browser (Standard: `localhost:8501`). Ticker in der Sidebar eingeben und Parameter anpassen.

## Hinweis

Wörtlich aus der App selbst:

> "Auch diese Version ist keine sichere Vorhersage. Sie erkennt historische Cycle-Strukturen, Stabilität und Phasenqualität und projiziert daraus mögliche Zeitfenster."

Dieses Projekt dient ausschließlich der eigenen Analyse und dem Lernen im Bereich Signalverarbeitung/Zeitreihenanalyse — **keine Finanz- oder Anlageberatung**.

---

*Persönliches Projekt, entstanden aus Interesse an Finanzmärkten und digitaler Signalverarbeitung.*
