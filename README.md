# SSHH: A Code Base for Combining SSH and Harper Models

Dieses Repository enthält den Code zur numerischen Untersuchung des kombinierten Modells aus Harper- und SSH-Modell:

1. **Harper-Modell:**
Das Harper-Modell (oder Aubry-André-Modell oder AAH-Modell) zeichnet sich durch einen Phasenübergang hinsichtlich der Lokalisierung der Zustände aus (Metall-Isolator).
2. **SSH-Modell:**
Das Su-Schrieffer-Heeger-Modell Modell zeichnet sich durch einen Phasenübergang hinsichtlich des Auftretens von Randzuständen aus, welche mit einer topologischen Größe des Balks zusammenhängen (Topologischer Isolator).

## Aufbau des Repositories

### Ordner

- `python/`: Python-Code zur Berechnung der Modelle.
- `scripts/`: Skripte für die CI/CD Pipeline und darüber hinaus. [Siehe unten](#skripte).
- `test/`: Tests für den Python-Code.

Folgende Ordner werden während der Berechnungen und Tests erzeugt: `data/`, `htmlcov/` und einige Unterordner in o. g. Ordnern.

### Skripte

Alle Skripte liegen im `scripts` Ordner und werden mit [just](https://github.com/casey/just) ausgeführt oder sind direkt im `justfile` notiert.
Tippe `just`, um alle verfügbaren Skripte anzeigen zu lassen.
just versucht, eine Datei `.env` zu laden, in der Umgebungsvariablen gesetzt werden können. Minimal sollte hier die Variable `PYTHONPATH` gesetzt werden, um den Python-Code zu finden:

    PYTHONPATH=python

just sollte via der offiziellen Paketquellen installierbar sein. Für ältere Debian/Ubuntu-Versionen steht das Skript `scripts/setup_just.sh` zur Verfügung.