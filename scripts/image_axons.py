from synthspline.cli import ImageApp
from synthspline.imagezoo import SynthAxonImage

ImageApp(
    SynthAxonImage(),
    ['label', 'prob'],
    ['image', 'prob'],
).run()
