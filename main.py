import kivy
from kivy.app import App
import kivy.core.text
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
import cv2
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
import numpy as np
import Recognition
import CreateEmbedVector
import PIL

Builder.load_string('''

<QrtestHome>:
    id:  qrtest
    orientation: "vertical"

    BoxLayout:
        id: Header
        orientation: "horizontal"
        height: 50

        size_hint_y: None

        canvas.before:
            Color:
                rgba: (0.25, 0.36, 1, 1)

            Rectangle:
                size: self.size
                pos: self.pos    

        Label:
            anchor_x: 'center'
            anchor_y: 'center'
            text: "Attendance Check"

        Button:
            id: ListPeople
            size_hint: 0.15, 0.8
            size_hint_max_x: 60
            text: "DS" 
            on_press: app.root.manager.current = 'second_screen'

    KivyCamera:
        id: qrcam

    BoxLayout:
        id: ControlButton
        orientation: "horizontal"
        height: 50
        size_hint_y: None

        Button:
            id: butt_start
            size_hint: 0.5,1
            text: "Start"
            on_press: qrtest.dostart()

        Button:
            id: butt_exit
            text: "Register"
            size_hint: 0.5,1
            on_press: qrtest.capture()

<AttendanceList>:
    id:  AttList
    orientation: "vertical"

    BoxLayout:
        id: Header1
        orientation: "horizontal"
        height: 50
        size_hint_y: None

        canvas.before:
            Color:
                rgba: (0.25, 0.36, 1, 1)

            Rectangle:
                size: self.size
                pos: self.pos    

        Label:
            anchor_x: 'center'
            anchor_y: 'center'
            text: "Attendance Check"

<First@Screen>:
    QrtestHome
<Second@Screen>:
    AttendanceList                                           
''')

model = CreateEmbedVector.Model()
find_match  = Recognition.Recognition()
find_match.update_List()

class First(Screen):
    pass

class Second(Screen):
    pass

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
            if box != None:
                cv2.rectangle(frame,(box[0], box[1]), (box[2], box[3]),thickness=1 ,color=(0,255,0))

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

    def capture(self):
        camera = self.ids['qrcam'].texture
        h, w = camera.height, camera.width
        im = np.frombuffer(camera.pixels, np.uint8)
        im = im.reshape(h, w, 4)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

        box, det_im, _ = model.detect(im, True, down_sample=1)
        if det_im == None:

            return

        embed_vec = model.create_vector(det_im)
        matches = find_match.Best_match(embed_vec)

        #This face MAY belong to a stranger
        if matches == -1:
            #pop up: Unrecognized face, need register
            print('who is this?')
            pass
        else:
            #tick pass for user,
            id = CreateEmbedVector.convert_i_to_id(matches)
            print(find_match.list_id[matches]['name'])
            pass


    def open_AttendanceList(self):
        pass

class AttendanceList(BoxLayout):
    pass

class qrtestApp(App):

    def build(self):
        Window.clearcolor = (.4,.4,.4,1)
        Window.size = (400, 500)
        sm =ScreenManager()
        sm.add_widget(First(name='first_screen'))
        sm.add_widget(Second(name='second_screen'))
        #homeWin = QrtestHome()
        #homeWin.init_qrtest()
        return sm

    def on_stop(self):
        global capture
        if capture:
            capture.release()
            capture = None




qrtestApp().run()