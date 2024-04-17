from vesselsynth.cli import ImageApp
from vesselsynth.imagezoo import SynthAxonImage

ImageApp(SynthAxonImage, ['label', 'prob'], ['image', 'prob']).run()
