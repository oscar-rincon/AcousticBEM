import numpy as np

def wavenumberToFrequency(k, c=344.0):
    """
    Convert wavenumber (k) to frequency (f).

    Parameters:
        k (float): Wavenumber in radians per meter.
        c (float, optional): Speed of sound in meters per second. Default is 344.0 m/s.

    Returns:
        float: Frequency in Hertz (Hz).
    """
    return 0.5 * k * c / np.pi

def frequencyToWavenumber(f, c=344.0):
    """
    Convert frequency (f) to wavenumber (k).

    Parameters:
        f (float): Frequency in Hertz (Hz).
        c (float, optional): Speed of sound in meters per second. Default is 344.0 m/s.

    Returns:
        float: Wavenumber in radians per meter.
    """
    return 2.0 * np.pi * f / c

def soundPressure(k, phi, t=0.0, c=344.0, density=1.205):
    """
    Calculate sound pressure as a complex value.

    Parameters:
        k (float): Wavenumber in radians per meter.
        phi (complex): Acoustic potential.
        t (float, optional): Time in seconds. Default is 0.0.
        c (float, optional): Speed of sound in meters per second. Default is 344.0 m/s.
        density (float, optional): Air density in kilograms per cubic meter. Default is 1.205 kg/m³.

    Returns:
        np.complex64: Sound pressure as a complex value.
    """
    angularVelocity = k * c
    return (1j * density * angularVelocity * np.exp(-1.0j * angularVelocity * t) * phi).astype(np.complex64)

def SoundMagnitude(pressure):
    """
    Calculate the sound magnitude in decibels (dB).

    Parameters:
        pressure (complex): Sound pressure.

    Returns:
        float: Sound magnitude in decibels (dB).
    """
    return np.log10(np.abs(pressure / 2e-5)) * 20

def AcousticIntensity(pressure, velocity):
    """
    Calculate the acoustic intensity.

    Parameters:
        pressure (complex): Sound pressure.
        velocity (complex): Particle velocity.

    Returns:
        float: Acoustic intensity in watts per square meter (W/m²).
    """
    return 0.5 * (np.conj(pressure) * velocity).real

def SignalPhase(pressure):
    """
    Calculate the phase of the signal in radians.

    Parameters:
        pressure (complex): Sound pressure.

    Returns:
        float: Phase of the signal in radians.
    """
    return np.arctan2(pressure.imag, pressure.real)