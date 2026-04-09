"""Variable definitions and metadata for ocean and wave data."""

from typing import Any, Dict, Optional

from blue_ml.dataprocessing.categories import (
    Magnitude,
    ModelScope,
    WaveFeature,
    WaveSource,
    WindHeight,
)

wvs = ModelScope.Waves
atm = ModelScope.Atmosphere
ocn = ModelScope.Ocean

swh = Magnitude.SignWaveHeight
wvd = Magnitude.WaveDirection
wvp = Magnitude.WavePeriod
wds = Magnitude.WindSpeed
wdd = Magnitude.WindDirection
crs = Magnitude.CurrentSpeed
crd = Magnitude.CurrentDirection
wtl = Magnitude.WaterLevel
arp = Magnitude.AirPressure
dsd = Magnitude.DirectionalStdDev

wnd = WaveSource.WindSea
swl = WaveSource.Swell
cmb = WaveSource.Combined

mea = WaveFeature.Mean
pea = WaveFeature.Peak
zec = WaveFeature.ZeroCrossing

ten = WindHeight.Ten
hun = WindHeight.Hundred

variable_dictionary: Dict[str, Dict[str, Optional[Any]]] = {
    "Air Pressure at Mean Sea Level (P_{Air})": {
        "scope": atm,
        "magnitude": arp,
        "source": None,
        "feature": None,
        "height": None,
    },
    "Current Direction (CD)": {
        "scope": ocn,
        "magnitude": crd,
        "source": None,
        "feature": None,
        "height": None,
    },
    "Current Speed (CS)": {
        "scope": ocn,
        "magnitude": crs,
        "source": None,
        "feature": None,
        "height": None,
    },
    "Directional Standard Deviation (DSD)": {
        "scope": wvs,
        "magnitude": dsd,
        "source": cmb,
        "feature": None,
        "height": None,
    },
    "Directional Standard Deviation - Swell (DSD_{Swell})": {
        "scope": wvs,
        "magnitude": dsd,
        "source": swl,
        "feature": None,
        "height": None,
    },
    "Directional Standard Deviation - Wind-Sea (DSD_{Sea})": {
        "scope": wvs,
        "magnitude": dsd,
        "source": wnd,
        "feature": None,
        "height": None,
    },
    "Mean Wave Direction (MWD)": {
        "scope": wvs,
        "magnitude": wvd,
        "source": cmb,
        "feature": mea,
        "height": None,
    },
    "Mean Wave Direction - Swell (MWD_{Swell})": {
        "scope": wvs,
        "magnitude": wvd,
        "source": swl,
        "feature": mea,
        "height": None,
    },
    "Mean Wave Direction - Wind-Sea (MWD_{Sea})": {
        "scope": wvs,
        "magnitude": wvd,
        "source": wnd,
        "feature": mea,
        "height": None,
    },
    "Mean Wave Period (T_{01})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": cmb,
        "feature": mea,
        "height": None,
    },
    "Mean Wave Period - Swell (T_{01,Swell})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": swl,
        "feature": mea,
        "height": None,
    },
    "Mean Wave Period - Wind-Sea (T_{01,Sea})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": wnd,
        "feature": mea,
        "height": None,
    },
    "Peak Wave Direction (PWD)": {
        "scope": wvs,
        "magnitude": wvd,
        "source": cmb,
        "feature": pea,
        "height": None,
    },
    "Peak Wave Direction - Swell (PWD_{Swell})": {
        "scope": wvs,
        "magnitude": wvd,
        "source": swl,
        "feature": pea,
        "height": None,
    },
    "Peak Wave Direction - Wind-Sea (PWD_{Sea})": {
        "scope": wvs,
        "magnitude": wvd,
        "source": wnd,
        "feature": pea,
        "height": None,
    },
    "Peak Wave Period (T_{p})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": cmb,
        "feature": pea,
        "height": None,
    },
    "Peak Wave Period - Swell (T_{p,Swell})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": swl,
        "feature": pea,
        "height": None,
    },
    "Peak Wave Period - Wind-Sea (T_{p,Sea})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": wnd,
        "feature": pea,
        "height": None,
    },
    "Sign. Wave Height (H_{m0})": {
        "scope": wvs,
        "magnitude": swh,
        "source": cmb,
        "feature": None,
        "height": None,
    },
    "Sign. Wave Height - Swell (H_{m0,Swell})": {
        "scope": wvs,
        "magnitude": swh,
        "source": swl,
        "feature": None,
        "height": None,
    },
    "Sign. Wave Height - Wind-Sea (H_{m0,Sea})": {
        "scope": wvs,
        "magnitude": swh,
        "source": wnd,
        "feature": None,
        "height": None,
    },
    "Water Level (WL)": {
        "scope": ocn,
        "magnitude": wtl,
        "source": None,
        "feature": None,
        "height": None,
    },
    "Wind Direction at 100m (WD_{100})": {
        "scope": atm,
        "magnitude": wdd,
        "source": None,
        "feature": None,
        "height": hun,
    },
    "Wind Direction at 10m (WD_{10})": {
        "scope": atm,
        "magnitude": wdd,
        "source": None,
        "feature": None,
        "height": ten,
    },
    "Wind Speed at 100m (WS_{100})": {
        "scope": atm,
        "magnitude": wds,
        "source": None,
        "feature": None,
        "height": hun,
    },
    "Wind Speed at 10m (WS_{10})": {
        "scope": atm,
        "magnitude": wds,
        "source": None,
        "feature": None,
        "height": ten,
    },
    "Wind Speed at 10m - Adjusted for wave model forcing (WS_{10})": {
        "scope": atm,
        "magnitude": wds,
        "source": None,
        "feature": None,
        "height": ten,
    },
    "Zero-crossing Wave Period (T_{02})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": swl,
        "feature": zec,
        "height": None,
    },
    "Zero-crossing Wave Period - Swell (T_{02,Swell})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": swl,
        "feature": zec,
        "height": None,
    },
    "Zero-crossing Wave Period - Wind-Sea (T_{02,Sea})": {
        "scope": wvs,
        "magnitude": wvp,
        "source": wnd,
        "feature": zec,
        "height": None,
    },
}
