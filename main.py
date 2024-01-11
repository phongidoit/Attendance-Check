import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from facenet_pytorch import MTCNN
from kivy.uix.camera import Camera
import time
import random

kivy.require('1.9.0')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MyRoot(BoxLayout):
	def __init__(self):
		super(MyRoot, self).__init__()

	def generate_number(self):
		self.random_number.text = str(random.randint(1,100))

class NeuralRandom(App):
	def build(self):
		return MyRoot()

class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")

neuralRandom = NeuralRandom()
neuralRandom.run()