from synthspline.cli import ImageApp
from synthspline.imagezoo import SynthVesselImage

ImageApp(
    SynthVesselImage(),
    ['label'],
    ['image', 'reference'],
).run()
