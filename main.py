import numpy as np
from scipy.signal import resample, savgol_filter
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Category10
from bokeh.models import Button, ColumnDataSource, Slider, Div, FileInput
from ZweikanalAnalyseClass import ZweikanalAnalyse
import base64
import shutil
# import spectacoular
import os

# Speichert den aktuellen Pfad (auf deinem Rechner) von main.py
# nötig , weil sphinx von /docs läuft und relative Pfade
# zu Problemen führen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_all(signal1Path = os.path.join(BASE_DIR, "Audiosignale", "signal1.wav"), signal2Path = os.path.join(BASE_DIR, "Audiosignale", "signal2.wav")):
    '''
    Führt die gesamte Analyse durch und gibt ein ZweikanalAnalyse-Objekt zurück.
    Args:
        signal1Path (str): Pfad zur WAV-Datei des ersten Signals.
        signal2Path (str): Pfad zur WAV-Datei des zweiten Signals.
    Returns:
        ZweikanalAnalyse: Ein Objekt der Klasse ZweikanalAnalyse mit den berechneten Werten.
    '''
    # Lade die Signale
    ts,fs = ZweikanalAnalyse.loadSignalWAV(signal1Path, signal2Path)
    sig1 = ts.data[:,0]
    sig2 = ts.data[:,1]

    # Downsampling der Signale
    downsampling_factor = slider_downsampling.value
    fsReduced           = ts.sample_freq/downsampling_factor
    nSampleReduced      = int(ts.num_samples/downsampling_factor)
    sig1Reduced         = resample(sig1, nSampleReduced)
    sig2Reduced         = resample(sig2, nSampleReduced)

    Analyse = ZweikanalAnalyse(sig1Reduced,sig2Reduced,fsReduced)               # ZweikanalAnalyse Objekt initalisieren
    Analyse.computeCorrelations()       # Auto- und Kreuzkorrelation berechnen
    Analyse.computePSD_CSD()            # Auto- und Kreuzleistungsspektren berechnen
    Analyse.computeFrequencyResponse()  # Übertragungsfunktion berechnen
    Analyse.computeCoherence()          # Kohärenz berechnen
    Analyse.computeImpulseResponse()    # Impulsantwort berechnen
    Analyse.time_axis = np.arange(len(Analyse.impulse_response))/Analyse.fs     # Zeitachse für die Impulsantwort
    Analyse.lags_sec = Analyse.correlationLags / Analyse.tsAcoular.sample_freq  # Umwandlung der Lags in Sekunden
    return Analyse

def smooth(data, window_length=21, polyorder=3):
    '''
    Glättet die Daten mit einem Savitzky-Golay-Filter.
    Args:
        data (array-like): Die zu glättenden Daten.
        window_length (int): Länge des Fensters für den Savitzky-Golay-Filter.
        polyorder (int): Ordnung des Polynoms für den Savitzky-Golay-Filter.
    Returns:
        array-like: Die geglätteten Daten.
    '''
    if len(data) < window_length:
        window_length = max(3, len(data) // 2 * 2 + 1)
    return savgol_filter(data, window_length, polyorder)

# header
header = Div(
    text =
    """<h1 style="font-size:2.5em;">
        Zweikanal-Analysator Dashboard
    </h1>
    <p style="font-size:1.3em;">
        Interaktive Visualisierung von Kreuzleistungsspektren, Kreuzkorrelation, Impulsantwort, Übertragungsfunktion und Kohärenz von zwei Signalen.
    </p>""",
    sizing_mode="stretch_width",
)

# Signalauswahl
file_input_0 = FileInput(accept=".wav")
file_input_1 = FileInput(accept=".wav")

def save_uploaded_file(file_input, filename):
    '''
    Speichert die hochgeladene Datei in einem bestimmten Verzeichnis.
    Args:
        file_input (FileInput): Das Bokeh FileInput Widget.
        filename (str): Der Pfad, unter dem die Datei gespeichert werden soll.
    '''
    if file_input.value:
        # Decode base64 and save to file
        file_bytes = base64.b64decode(file_input.value)
        with open(filename, "wb") as f:
            f.write(file_bytes)

# Callbacks für die FileInputs
def on_file_input_0_change(attr, old, new):
    save_uploaded_file(file_input_0, os.path.join(BASE_DIR, "Audiosignale", "signal1_aktuell.wav"))

def on_file_input_1_change(attr, old, new):
    save_uploaded_file(file_input_1, os.path.join(BASE_DIR, "Audiosignale", "signal2_aktuell.wav"))

file_input_0.on_change("value", on_file_input_0_change)
file_input_1.on_change("value", on_file_input_1_change)

# Downsampling Slider
slider_downsampling = Slider(title="Downsampling Faktor", start=1, end=20, value=1, step=1)

# Button zum Starten der Analyse
my_button = Button(label="Analyse starten", button_type="success")
my_button.on_click(lambda: on_button_click())

# kopiere die Signale signal1.wav und signal2.wav und speichere sie als signal1_aktuell.wav und signal2_aktuell.wav
shutil.copy(os.path.join(BASE_DIR, "Audiosignale", "signal1.wav"), os.path.join(BASE_DIR, "Audiosignale", "signal1_aktuell.wav"))
shutil.copy(os.path.join(BASE_DIR, "Audiosignale", "signal2.wav"), os.path.join(BASE_DIR, "Audiosignale", "signal2_aktuell.wav"))

# Pfad zu den Audiosignalen
signal1Path = os.path.join(BASE_DIR, "Audiosignale", "signal1_aktuell.wav")
signal2Path = os.path.join(BASE_DIR, "Audiosignale", "signal2_aktuell.wav")

# Berechnung der Analyse für die default-Signale
Analyse = calculate_all(signal1Path, signal2Path)

# Initialisierung der Datenquellen
power_source_abs = ColumnDataSource(data=dict(
    freqs=Analyse.freqs,
    auto1=np.abs(Analyse.psd1),                # Leistungsspektrum NICHT glätten!
    auto2=np.abs(Analyse.psd2),
    cross=np.abs(Analyse.csd)
))
power_source_phase = ColumnDataSource(data=dict(
    freqs=Analyse.freqs,
    cross=smooth(np.angle(Analyse.csd))        # Kreuzphase glätten
))
correlation_source = ColumnDataSource(data=dict(
    lags_sec=Analyse.lags_sec,
    auto_corr1=smooth(Analyse.auto_corr1),     # Korrelationen glätten
    auto_corr2=smooth(Analyse.auto_corr2),
    cross_corr=smooth(Analyse.cross_corr)
))
transfer_source = ColumnDataSource(data=dict(
    freqs=Analyse.freqs,
    H=smooth(np.abs(Analyse.H))                # Übertragungsfunktion glätten
))
impulse_source = ColumnDataSource(data=dict(
    time_axis=Analyse.time_axis,
    h=smooth(np.real(Analyse.impulse_response))# Impulsantwort glätten
))
coherence_source = ColumnDataSource(data=dict(
    freqs=Analyse.freqs,
    coh=smooth(np.abs(Analyse.coherence))      # Kohärenz glätten
))

# Bokeh Plots
power_fig_abs = figure(title="Leistungs- und Kreuzspektren (Absolutwerte)", x_axis_label="Frequenz [Hz]", y_axis_label="Amplitude", y_axis_type="log")
power_fig_phase = figure(title="Leistungs- und Kreuzspektren (Phase)", x_axis_label="Frequenz [Hz]", y_axis_label="Phase [rad]", y_axis_type="linear")
correlation_fig = figure(title="Auto- und Kreuzkorrelation", x_axis_label="Verzögerung [s]", y_axis_label="Korrelation [1]", x_range=(-0.01, 0.01))
impulse_fig = figure(title="Impulsantwort", x_axis_label="Zeit [s]", y_axis_label="Amplitude [1]")
transfer_fig = figure(title="Übertragungsfunktion", x_axis_label="Frequenz [Hz]", y_axis_label="Amplitude [1]", x_axis_type="log")
coherence_fig = figure(title="Kohärenz", x_axis_label="Frequenz [Hz]", y_axis_label="Kohärenz [1]")

# Linien zu den Plots hinzufügen
power_fig_abs.line('freqs', 'auto1', source=power_source_abs, legend_label="Leistungsspektrum Signal 1", color=Category10[5][0])
power_fig_abs.line('freqs', 'auto2', source=power_source_abs, legend_label="Leistungsspektrum Signal 2", color=Category10[5][1])
power_fig_abs.line('freqs', 'cross', source=power_source_abs, legend_label="Kreuzleistungsspektrum", color=Category10[5][2])

correlation_fig.line('lags_sec', 'auto_corr1', source=correlation_source, legend_label="Autokorrelation Signal 1")
correlation_fig.line('lags_sec', 'auto_corr2', source=correlation_source, legend_label="Autokorrelation Signal 2", color=Category10[5][1])
correlation_fig.line('lags_sec', 'cross_corr', source=correlation_source, legend_label="Kreuzkorrelation", color=Category10[5][2])

power_fig_phase.line('freqs', 'cross', source=power_source_phase, legend_label="Kreuzphase", color=Category10[5][2])
transfer_fig.line('freqs', 'H', source=transfer_source, legend_label="Magnitude", color=Category10[5][0])
impulse_fig.line('time_axis', 'h', source=impulse_source, legend_label="Impulsantwort", color=Category10[5][0])
coherence_fig.line('freqs', 'coh', source=coherence_source, legend_label="Kohärenz", color=Category10[5][0])

# Einstellung der plots
for fig in [power_fig_abs, transfer_fig, impulse_fig, coherence_fig, correlation_fig]:
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_right"
    fig.grid.grid_line_alpha = 0.3

def on_button_click():
    '''
    Callback-Funktion, die ausgeführt wird, wenn der Button geklickt wird.
    Sie lädt die aktuell ausgewählten Signale und berechnet alle Analysewerte neu.
    '''
    # Lade die aktuell ausgewählten Signale
    signal1Path = os.path.join(BASE_DIR, "Audiosignale", "signal1_aktuell.wav")
    signal2Path = os.path.join(BASE_DIR, "Audiosignale", "signal2_aktuell.wav")
    Analyse = calculate_all(signal1Path,signal2Path)

    time_axis = np.arange(len(Analyse.impulse_response))/Analyse.fs
    lags_sec = Analyse.correlationLags / Analyse.tsAcoular.sample_freq

    # Update der Datenquellen
    power_source_abs.data = dict(
        freqs=Analyse.freqs,
        auto1=np.abs(Analyse.psd1),                # Leistungsspektrum NICHT glätten!
        auto2=np.abs(Analyse.psd2),
        cross=np.abs(Analyse.csd)
    )
    power_source_phase.data = dict(
        freqs=Analyse.freqs,
        cross=smooth(np.angle(Analyse.csd))        # Kreuzphase glätten
    )
    correlation_source.data = dict(
        lags_sec=lags_sec,
        auto_corr1=smooth(Analyse.auto_corr1),     # Korrelationen glätten
        auto_corr2=smooth(Analyse.auto_corr2),
        cross_corr=smooth(Analyse.cross_corr)
    )
    transfer_source.data = dict(
        freqs=Analyse.freqs,
        H=smooth(np.abs(Analyse.H))                # Übertragungsfunktion glätten
    )
    impulse_source.data = dict(
        time_axis=time_axis,
        h=smooth(np.real(Analyse.impulse_response))# Impulsantwort glätten
    )
    coherence_source.data = dict(
        freqs=Analyse.freqs,
        coh=smooth(np.abs(Analyse.coherence))      # Kohärenz glätten
    )

# Layout der Bokeh-App
power_fig_abs.legend.title = "Leistungsspektren"
correlation_fig.legend.title = "Korrelationen"
layout = column(
    header,
    Div(text="<h2>Signalauswahl</h2>", styles={"font-size": "1.5em"}),
    row(file_input_0, file_input_1),
    Div(text="<h2>Downsampling</h2>", styles={"font-size": "1.5em"}),
    slider_downsampling,
    my_button,
    Div(text="<h2>Analyseergebnisse</h2>", styles={"font-size": "1.5em"}),
    Div(text="<h3>Leistungs- und Kreuzspektren</h3>", styles={"font-size": "1.5em"}),
    row(power_fig_abs, power_fig_phase),
    Div(text="<h3>weitere Größen</h3>", styles={"font-size": "1.5em"}),
    row(correlation_fig, coherence_fig),
    row(impulse_fig, transfer_fig),
    
    sizing_mode = "stretch_width"
)

curdoc().add_root(layout)

#####################################################################################

# RUN WITH THIS EXPRESSION IN CONSOLE:

#       bokeh serve --show main.py


# This will start a Bokeh server and open the dashboard in your web browser.
# Make sure to have the required libraries installed:

#       pip install bokeh numpy scipy matplotlib soundfile shutil


# The dashboard will allow you to upload two audio files, perform the analysis,
# and visualize the results interactively.
# You can also adjust the downsampling factor to see how it affects the results.
# for testing purposes, you can use the default audio files in the Audiosignale directory.
# The file-pairs are:
#       signal1.wav and signal2.wav
#       noise_a.wav and noise_b.wav
#       blue_audio.wav and blue_audio_muffled.wav
# The default files are already copied to the Audiosignale directory.

#####################################################################################