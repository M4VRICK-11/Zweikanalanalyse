import numpy as np
import acoular as ac
from scipy.signal import correlate
import soundfile as sf

class ZweikanalAnalyse:
    def __init__(self, signal1, signal2, fs):
        '''
        Instanziert ein Objekt der Signalanalyse-Klasse `Zweikanalanalyse`
        mit den folgenden Parametern.

        Args:
            signal1 (array): Der zeitdiskrete Signal 1 vom Kanal 1
            signal2 (array): Der zeitdiskrete Signal 2 vom Kanal 2
            fs (int): Abtastrate der Signale in Hz
        '''
        self.signal1    = signal1
        self.signal2    = signal2
        self.fs         = fs
        self.duration   = len(signal1) / fs
        self.tsAcoular  = self.build_tsAcoularObject()
        self.auto_corr1 = None
        self.auto_corr2 = None
        self.cross_corr = None
        self.freqs      = None
        self.psd1       = None
        self.psd2       = None
        self.csd        = None
        self.H          = None
        self.impulse_response   = None
        self.block_size = None
        self.correlationLags    = None
        self.coherence  = None
        self.time_axis  = None
        self.lags_sec   = None

    def createTestSignal(fs=48000,duration=2.0):
        '''
        Die Funktion erzeugt zwei Testsignale und speichert sie als WAV-Dateien.

        Signal 1: Reiner Sinus bei 440 Hz

        Signal 2: Gleicher Sinus + kleine Phase + gaußeschen Rauschen

        Args:
            fs (int): Abtastrate in Hz
            duration (float): Dauer des Signals in Sekunden
        '''
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        # Signal 1: Reiner Sinus bei 440 Hz
        freq    = 440  # Hz
        signal1 = 0.5 * np.sin(2 * np.pi * freq * t)
        # Signal 2: Gleicher Sinus + kleine Phase + Rauschen
        phase_shift = np.pi / 4  # 45°
        signal2     = 0.5 * np.sin(2 * np.pi * freq * t + phase_shift) + 0.05 * np.random.randn(len(t))
        # WAV-Dateien schreiben
        sf.write("signal1.wav", signal1, fs)
        sf.write("signal2.wav", signal2, fs)

    def loadSignalWAV(signal1Path, signal2Path):
        '''
        Lädt die aufbereitete Signale und speichert sie zusammen mit deren Abtastfrequenz in einem Objekt der Klasse Acoular.

        Die Funktion überprüft, dass die Signallängen überreinstimmen gibt neben das Acoular-Objekt auch die Abtastfrequenz vereinzelt zurück.

        Args:
            signal1Path (string)   :  Pfad Eingagnssignale
            signal2Path (string)   :  Pfad Ausgangssignale

        Returns:
            ac.TimeSamples, int : **Acoular-Objekt**: mit beiden Signalen im Zeitbereich und deren Abtastfrequenz. **fsS1**: die Abtastfrequenz der Signale (separat).
    
        '''
        # lädt Inhalt der Signale und Abtastrate
        dataS1, fsS1 = sf.read(signal1Path)
        dataS2, fsS2 = sf.read(signal2Path)

        assert fsS1 == fsS2 # Überprüft ob Abtastraten gleich sind 
        min_len = min(len(dataS1), len(dataS2))
        data    = np.stack([dataS1[:min_len], dataS2[:min_len]], axis=1)
        return ac.TimeSamples(data=data, sample_freq=fsS1), fsS1

    def build_tsAcoularObject(self):
        '''
        Speichert die zwei Signale sowie deren Abtastfrequenz in einem
        Acoular-Objekt. 

        Der Acoular-Objekt sorgt für Kompatibilität mit den Acoular Methoden und Funktionen, die für weitere Analysen notwendig sind.
        
        Note:
            Erforderlich für die Konvertierung sind:

            - `self.signal1`  (ndarray) : Der zeitdiskrete Audiosignal vom Kanal 1.
            - `self.signal2`  (ndarray) : Der zeitdiskrete Audiosignal vom Kanal 2.
            - `self.fs`       (int)     : Die Abtastfrequenz von beiden Audiosignalen.

            Gibt zurück:

            - Acoular-Objekt  (ac.TimeSample): Enthält die zeitdiskreten Signaldaten und die Abtastfrequenz.
        '''
        data = np.stack([self.signal1[:], self.signal2[:]], axis=1)
        return ac.TimeSamples(data=data, sample_freq=self.fs)
    
    def computeCorrelations(self):
        '''
        Berechnet die Auto- und Kreuzkorrelation von zwei zu analysierenden zeitdiskreten Signalen.

        Für die Berechnung der Auto- und Kreuzkorrelation wird die `scipy.signal.correlate()` Funktion
        eingesetzt.
        
        Note:
            Erforderlich für die Berechnung ist:

            - `self.tsAcoular` (acoular.TimeSamples) : Ein Acoular-Objekt, das die zeitdiskreten Signaldaten und deren Abtastfrequenz enthält.
            
            Die Methode berechnet:

            - `lags`        (array): Array der Zeitverzögerungen (Samples)
            - `auto_corr1`  (array): Autokorrelation von Kanal 1
            - `auto_corr2`  (array): Autokorrelation von Kanal 2
            - `cross_corr`  (array): Kreuzkorrelation zwischen Kanal 1 und 2

            Die Ergebnisse werden mit den `set_correlation_lags()`, `set_autocorrelations()` und `set_cross_correlation()` Methoden gespeichert. (Siehe die jeweilige Dokumentation)
        '''
        # Daten aus ts Objekt extrahieren
        sig1 = self.signal1
        sig2 = self.signal2

        # Correlate-Objekt initialisieren
        auto_corr1 = correlate(sig1, sig1, mode='full')
        auto_corr2 = correlate(sig2, sig2, mode='full')
        cross_corr = correlate(sig1, sig2, mode='full')

        # Lags berechnen
        lags = np.arange(- self.tsAcoular.numsamples + 1,  self.tsAcoular.numsamples)
        
        # Werte speichern
        self.set_autocorrelations( auto_corr1, auto_corr2)
        self.set_cross_correlation( cross_corr)
        self.set_correlation_lags(lags)

    def set_correlation_lags(self, lags):
        '''
        Speichert die Lags der Korrelationen. Diese dienen der Erstellung einer Zeitachse zur Visualisierung der Auto- und Kreuzkorrelationfunktionen.

        Args:
            lags (array): Array der Zeitverzögerungen in Samples

        Note:
            Die Methode speichert die Zeitverzögerungen (`lags`) zwischen den Signalen in:

            - `self.correlationLags` (ndarray): Objektattribut zu den Korrelations-Lags
        '''
        self.correlationLags = lags
        
    def set_autocorrelations(self, ac1, ac2):
        '''
        Speichert die Autokorrelationen der Signale.

        Args:
            ac1 (array): Autokorrelation des ersten Signals
            ac2 (array): Autokorrelation des zweiten Signals
        
        Note:
            Die Methode speichert die berecheten Autokorrelationen (`ac1` und `ac2`) in:

            - `self.auto_corr1` (ndarray): Objektattribut zur berechneten Autokorrelation vom Signal 1
            - `self.auto_corr2` (ndarray): Objektattribut zur berechneten Autokorrelation vom Signal 2

        '''
        self.auto_corr1 = ac1
        self.auto_corr2 = ac2

    def set_cross_correlation(self, cc):
        '''
        Speichert die Kreuzkorrelation der Signale.

        Args:
            cc (array): Kreuzkorrelation zwischen den beiden Signalen
        
        Note:
            Die Methode speichert die berecheten Kreuzkorrelation (`cc`) in:

            - `self.cross_corr` (ndarray): Objektattribut zur berechneten Kreuzkorrelation zwischen Signal 1 und 2
 
        '''
        self.cross_corr = cc
        
    def setField(self, fieldName, fieldValue):
        '''
        Speichert ein beliebiges Feld im ZweikanalAnalyse-Objekt.

        Diese Funktion dient der Erweiterbarkeit der Software, indem sie dem Programmierer erlaubt, weitere berechneten Größen zu speichern.

        Args:
            fieldName (str): Name des Feldes, das gespeichert werden soll.
            fieldValue: Wert des Feldes, das gespeichert werden soll.
        '''
        setattr(self, fieldName, fieldValue)
        
    def computePSD_CSD(self,block_size=512):
        '''
        Transformiert die zeitdiskreten Signale in den Frequenzbereich und berechnet die Auto- und Kreuzleistungsspektren.

        - Berechnet die Gesamtanzahl an Segmenten (`nBlocks`) durch die Teilung der Gesamtanzahl an Samples in den zeitdiskreten
          Signalen durch die festgelegte Segmentlänge (`Block_size`).

        - Instanziert einen RFFT Acoular-Generator (`acoular.spectra.RFFT`) mit den zeitdiskreten Signalen und deren Abtastfrequenz,
          um auf die realen FFT der Signale vorzubereiten.

        - Instanziert einen Acoular-Generator (`acoular.spectra.CrossPowerSpectra`), der auf den RFFT Generator aufbaut, zur Berechnung
          der Auto- und Kreuzleistungsspektren.

        - Instanziert einen Acoular-Generator (`acoular.process.Average`), der auf die anderen Generatoren aufbaut, um die Scharmittelung
          der berechneten Größen zu gewährleisten.

        - Erzeugt die Kreuzleistungsmatrix (`csmFlat`)

        - Extrahiert die Auto- und Kreuzleistungsspektren.

        Note:

            Erforderlich für die Berechnung:

            - `self.tsAcoular` (acoular.TimeSamples) : Enthält Daten der beiden zeitdiskreten Signale und deren Abtastfrequenz
            
            Die Methode berechnet:

            - `psd1` (array) : Das Autoleistungsspektrum vom Signal 1 (Kanal 1)
            - `psd2` (array) : Das Autoleistungsspektrum vom Signal 2 (Kanal 2)
            - `csd`  (array) : Das Kreuzleistungsspektrum von beiden Signalen

            Die Ergebnisse werden mit der `set_psd_csd()` Methode überarbeitet und gespeichert. (Siehe die jeweilige Dokumentation)
        '''
        # Anzahl Blöcke über die Ergebnisse gemittelt werden
        nBlocks     = int(self.tsAcoular.numsamples/block_size)
        # FFT
        fft         = ac.RFFT(source=self.tsAcoular, block_size=block_size)
        fft.scaling = 'none'
        # Bilde Auto-/Kreuzleistungsspektrum  
        cps         = ac.CrossPowerSpectra(source=fft)
        # Mittelwert für jeden Block
        avg         = ac.Average(source=cps, naverage=nBlocks)
        csmFlat     = next(avg.result(num=1))
        # Dimension anpassen so dass cms PSD und CPSD gesondert in einem array
        csmMatrix   = csmFlat.reshape(fft.numfreqs, self.tsAcoular.num_channels, self.tsAcoular.num_channels)
        psd1        = csmMatrix[:,0,0]
        psd2        = csmMatrix[:,1,1]
        csd         = csmMatrix [:,0,1]
        # Werte speichern
        self.set_psd_csd( fft.freqs, psd1, psd2, csd,block_size)
        
    def set_psd_csd(self, freqs, psd1, psd2, csd,block_size):
        '''
        Speichert die berechneten Autoeistungsspektren und das berechnete Kreuzleistungsspektrum von beiden Signalen.

        Die Funktion speichert ebenso die Frequenzachse zur Visualisierung der Spektren.

        Args:
            freqs (array): Frequenzachse
            psd1 (array): Leistungsspektrum des ersten Signals
            psd2 (array): Leistungsspektrum des zweiten Signals
            csd (array): Kreuzleistungsspektrum zwischen den beiden Signalen
            block_size (int): Größe der FFT-Blöcke
        
        Note:
            Die Methode speichert die berecheten Größen in:

            - `self.psd1` (ndarray): Objektattribut zum berechneten  Autoleistungsspektrum vom Signal 1
            - `self.psd2` (ndarray): Objektattribut zum berechneten  Autoleistungsspektrum vom Signal 2
            - `self.csd` (ndarray): Objektattribut zum berechneten  Kreuzleistungsspektrum der Signale
            - `self.freqs` (ndarray): Objektattribut zur erzeugten Frequenzachse für die Visualisierung der Auto- und Kreuzleistungsspektren.
        '''
        # self.windowSize = block_size
        self.freqs      = freqs
        self.psd1       = psd1
        self.psd2       = psd2
        self.csd        = csd

    def computeFrequencyResponse(self):
        '''
        Berechnet den Frequenzgang :math:`H(f)` aus der Kreuzleistungsmatrix.

        Der Frequenzgang ist nach dem H1-Schätzer wie folgt definiert als:

        .. math::

            H(f) = \\frac{C_{xy}(f)}{C_{xx}(f)}

        Dabei ist :math:`C_{xy}(f)` das Kreuzleistungsspektrum und
        :math:`C_{xx}(f)` das AutoLeistungsspektrum vom Signal 1 (Kanal 1).

        Note:
            Erforderlich für die Berechnung:

            - `self.csd`  (ndarray) : Kreuzleistungsspektrum der beiden Signale.
            - `self.psd1` (ndarray) : Autoeistungsspektrum des ersten Signals.

            Die Methode berechnet:

            - `H` (ndarray): deb Frequenzgang zum betrachteten Schallübertragungssystem.

            Das Ergebnis wird mit der `set_transfer_function()` Methode gespeichert. (Siehe Dokumentation)
        '''
        H = self.csd / self.psd1
        # Werte speichern
        self.set_transfer_function( H)

    def set_transfer_function(self, H):
        '''
        Speichert den Frequenzgang :math:`H(f)`.

        Args:
            H (array): der Frequenzgang :math:`H(f)` des betrachteten Schallübertragungssystems

        Note:
            Die Methode speichert die berecheten Größen in:

            - `self.H` (ndarray): Objektattribut zum berechneten Frequenzgang
        '''
        self.H = H

    def computeImpulseResponse(self):
        '''
        Rekonstruiert die Impulsantwort :math:`h(t)` im Zeitbereich über die inverse FFT des Frequenzgangs :math:`H(f)`.

        Diese Methode berechnet die Impulsantwort mit der inversen FFT Funktion von Numpy `np.fft.irfft()`.

        Die Impulsantwort ist definiert als:

        .. math::

            h(t) = \\frac{d}{dt} H(f)
        
        Note:
            Erforderlich für die Berechnung ist:

            - `self.H`  (ndarray) : Den berechneten Frequenzgang.

            Die Methode berechnet:

            - `h` (ndarray): Die rekonstruierte Impulsantwort vom betrachteten Schallübertragungssystem.

            Das Ergebnis wird mit der `set_impulse_response()` Methode gespeichert. (Siehe Dokumentation)
        '''
        h = np.fft.irfft(self.H)
        self.set_impulse_response( h)

    def set_impulse_response(self, h):
        '''
        Speichert die berechnete Impulsantwort h(t).
        
        Die Methode erzeugt auch eine Zeitachse für die Visualisierung des Ergebnisses.

        Args:
            h (array): Impulsantwort im Zeitbereich
        
        Note:
            Die Methode speichert die berechete Impulsantwort (`h`) in:

            - `self.impulse_response` (ndarray): Objektattribut zur rekonstruierten Impulsantwort

            Die Methode speichert die erzeugte Zeitachse für die spätere Visualisierung:

            - `self.time_axis` (ndarray): Objektattribut zur erzeugten Zeitachse für die Visualisierung der berechneten Impulsantwort.
        '''
        self.time_axis          = np.arange(len(h)) / self.fs
        self.impulse_response   = h
            
    def computeCoherence(self):
        '''
        Berechnet die Kohärenzfunktion zwischen den beiden Signalen.
        Die Kohärenz ist definiert als:

        .. math::

            \\gamma^2(f) = \\frac{\\left| C_{xy}(f) \\right|^2}{C_{xx}(f) \\cdot C_{yy}(f)}
        
        Dabei ist :math:`C_{xy}(f)` das Kreuzleistungsspektrum und
        :math:`C_{xx}(f)`, :math:`C_{yy}(f)` die jeweiligen AutoLeistungsspektren.

        Note:
            Erforderlich für die Berechnung sind:

            - `self.csd`  (ndarray) : Kreuzleistungsspektrum der beiden Signale.
            - `self.psd1` (ndarray) : Autoeistungsspektrum des ersten Signals.
            - `self.psd2` (ndarray) : Autoeistungsspektrum des zweiten Signals.

            Die Methode berechnet:

            - `coherence` (ndarray): Die berechnete Kohärenzfunktion.

            Das Ergebnis wird mit der `set_coherence()` Methode gespeichert. (Siehe Dokumentation)
        '''
        coherence = np.abs(self.csd)**2 / (self.psd1 * self.psd2)
        self.set_coherence(coherence)
        
    def set_coherence(self,coherence):
        '''
        Speichert die Kohärenzfunktion.

        Args:
            coherence (array): Frequenzabhängige Kohärenz zwischen den Signalen

        Note:
            Die Methode speichert die berechete Kohärenz (`coherence`) in:

            - `self.coherence` (ndarray): Objektattribut zur berechneten Kohärenz.

        '''
        self.coherence = coherence