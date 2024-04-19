from synthspline.cli import LabelApp
from synthspline.labelzoo import SynthVesselHiResMRI

LabelApp(SynthVesselHiResMRI, shape=128).run()
