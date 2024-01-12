import kivy
from kivy.app import App
import kivy.core.text
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
import cv2
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics.texture import Texture
import numpy as np
import CreateEmbedVector

Builder.load_string('''
<QrtestHome>:

    BoxLayout:
        orientation: "vertical"

        Label:
            height: 20
            size_hint_y: None
            text: 'Testing the camera'

        KivyCamera:
            id: qrcam

        BoxLayout:
            orientation: "horizontal"
            height: 20
            size_hint_y: None

            Button:
                id: butt_start
                size_hint: 0.5,2
                text: "start"
                on_press: root.dostart()

            Button:
                id: butt_exit
                text: "quit"
                size_hint: 0.5,2
                on_press: root.doexit()
''')

class KivyCamera(Image):

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.model = CreateEmbedVector.Model()

    def start(self, capture, fps=24):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):
        Clock.unschedule_interval(self.update)
        self.capture = None

    def update(self, dt):
        return_value, frame = self.capture.read()
        if return_value:
            texture = self.texture

            w, h = frame.shape[1], frame.shape[0]
            box, _, e = self.model.detect(frame,False,down_sample=8)
            print(box, e)

            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()


capture = None


class QrtestHome(BoxLayout):

    def init_qrtest(self):
        pass

    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)

    def doexit(self):
        global capture
        if capture != None:
            capture.release()
            capture = None
        EventLoop.close()


class qrtestApp(App):

    def build(self):
        Window.clearcolor = (.4,.4,.4,1)
        Window.size = (400, 300)
        homeWin = QrtestHome()
        homeWin.init_qrtest()
        return homeWin

    def on_stop(self):
        global capture
        if capture:
            capture.release()
            capture = None

qrtestApp().run()