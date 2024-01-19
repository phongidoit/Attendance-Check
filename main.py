import kivy
from kivy.app import App
import kivy.core.text
from kivy.graphics import Rectangle
from kivy.graphics import Color
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
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

#from android.permissions import request_permissions, Permission


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
            on_press: root.parent.manager.current = 'second_screen'

    KivyCamera:
        id: qrcam

    BoxLayout:
        id: ControlButton
        orientation: "vertical"
        size_hint_y: 0.3
        size_hint_y_min: 150
        
        BoxLayout:  
            id: nameField    
                 
            TextInput:
                id: newName
                hint_text: "New person's name"
                size_hint: 1, 1
                id: nameInput
                multiline: False
        
            Button:
                id: add_new
                text: "Add"
                disabled: True
                size_hint: 0.4, 1
                on_press:   qrtest.add_new()
                
        Button:
            id: checkIn
            text: "Check attendance"
            disabled: True
            size_hint: 1,1
            on_press: qrtest.capture()        
            
        Button:
            id: butt_start
            size_hint: 1, 1.5
            text: "Start/Stop Camera"
            on_press: qrtest.dostart()
        
<AttendanceList>:
    id:  AttList
    orientation: "vertical"
    minimum_size : root.parent.size 
    BoxLayout:
        id: Header1
        size_hint: 1, 0.15
        size_hint_max_y: 50
        orientation: "horizontal"
        #pos_hint: {"right":0.1,"top":0.5}
       
        canvas.before:
            Color:
                rgba: (0.25, 0.36, 1, 1)

            Rectangle:
                size: self.size
                pos: self.pos  
                  
        Button:
            id: butt_back
            size_hint: 0.15, 0.8
            
            text: "Back"
            size_hint_max_x: 65
            on_press: root.parent.manager.current="first_screen"
            
        Label:
            anchor_x: 'center'
            anchor_y: 'center'
            text: "Attendance Check"
            
  
            
        Button:
            id: load_list
            size_hint: 0.3, 0.8
            text: "Uncheck All"
            on_press: AttList.uncheck_all()  
            
    ScrollView: 
        id: ListAttendanceView
        do_scroll_x: False
        do_scroll_y: True
        scroll_timeout: 250
        scroll_distance: 20
        size_hint:(1,1) 
        
    Button:
        id: reload
        size_hint: 1, 0.2
        background_color: (0.15, 0.1, 0.8, 1)
        text: "Reload"
        on_press: AttList.update_cBox()                   

<First>:
    QrtestHome
<Second>:
    AttendanceList                                           
''')

model = CreateEmbedVector.Model()
find_match  = Recognition.Recognition()
find_match.update_List()

class First(Screen):
    pass

class Second(Screen):
    #Because of threading, on enter will run before inner child is load so I can't access ids
    #use clock to schedule the time
    #Got answer from: https://stackoverflow.com/questions/72174380/attribute-error-when-using-kivy-screen-manager-and-self-ids
    def on_enter(self, *args):
        Clock.schedule_once(self.run_stuff, 0.1)

    def run_stuff(self,  dt=0):
        print("IDs of screen 2: ", self.ids)

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

class Welcome_popup(Popup):
    def __init__(self,  person_name, **kwargs):
        super(Welcome_popup, self).__init__(**kwargs)

        self.size_hint = (0.8, 0.5)

        self.title="Welcome new person"
        self.content = BoxLayout(orientation='vertical')

        sMessage = "Hello, "+ person_name+"! Thank you for register!"
        self.message = Label(text=sMessage)
        self.accButton = Button(text="OK", size_hint_max_y=70)
        self.accButton.bind(on_press= self.dismiss )

        self.content.add_widget(self.message)
        self.content.add_widget(self.accButton)

class Message_popup(Popup):
    def __init__(self,  message, **kwargs):
        super(Message_popup, self).__init__(**kwargs)

        self.size_hint = (0.8, 0.5)

        self.title="Notification"
        self.content = BoxLayout(orientation='vertical')

        self.message = Label(text=message)
        self.accButton = Button(text="OK", size_hint_max_y=70)
        self.accButton.bind(on_press= self.dismiss )

        self.content.add_widget(self.message)
        self.content.add_widget(self.accButton)

class QrtestHome(BoxLayout):
    def init_qrtest(self):
        pass

    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)
        self.ids["add_new"].disabled = False
        self.ids["checkIn"].disabled = False

    def switch_cam(self, *largs):
        i=1
        capture = cv2.VideoCapture(i)
        self.ids.qrcam.start(capture)

    def doexit(self):
        global capture
        if capture != None:
            capture.release()
            capture = None
        EventLoop.close()

    def capture(self):
        if len(find_match.list_vector) == 0:
            mes = "Database is empty, add more people!"
            Empty_db_popup = Message_popup(mes)
            Empty_db_popup.open()
            return
        camera = self.ids['qrcam'].texture
        h, w = camera.height, camera.width
        im = np.frombuffer(camera.pixels, np.uint8)
        im = im.reshape(h, w, 4)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

        box, det_im, _ = model.detect(im, False, down_sample=1)
        if det_im == None:
            mes = "No face detected!"
            fail_det_popup = Message_popup(mes)
            fail_det_popup.open()
            return

        embed_vec = model.create_vector(det_im)[0]
        matches = find_match.Best_match(embed_vec)

        if matches == -1:
            mes = "Unable to recognize this face, try again!"
            stranger_popup = Message_popup(mes)
            stranger_popup.open()
            return
        else:
            #Ouput notification and update is_here status to both list and json file,
            self.ids['nameInput'].hint_text = "Hello "+ find_match.list_id[matches]['name']+"!"

            id = CreateEmbedVector.convert_i_to_id(matches)

            find_match.list_id[matches]['is_here'] = 1
            model.save_embed_vector(find_match.list_id)
            return


    def add_new(self):
        new_id = CreateEmbedVector.convert_i_to_id( int(find_match.list_id[-1]['id'])+1 )
        name = "Hello"
        camera = self.ids['qrcam'].texture
        h, w = camera.height, camera.width
        im = np.frombuffer(camera.pixels, np.uint8)
        im = im.reshape(h, w, 4)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

        box, det_im, _ = model.detect(im, False, down_sample=1)

        if det_im == None:
            mes = "No face detected!"
            fail_det_popup = Message_popup(mes)
            fail_det_popup.open()
            return

        name = self.ids['nameInput'].text
        if name == "":
            mes = "Type the name of this user before add!"
            empty_name_popup = Message_popup(mes)
            empty_name_popup.open()
            return

        self.ids['nameInput'].text = ""

        embed_vec = model.create_vector(det_im)

        find_match.list_id.append({"id":new_id, "name":name, "is_here": True})

        #Save new vector and is id
        model.save_embed_vector(find_match.list_id, embed_vec, new_id,name )

        find_match.update_List()
        self.popup = Welcome_popup(name)
        self.popup.open()

class Entry(BoxLayout):
    def __init__(self, usrid, name, is_here, **kwargs):
        super(Entry, self).__init__(**kwargs)
        self.id = usrid
        self.orientation = "horizontal"
        self.size_hint = (0.95, None)
        self.height = 100
        self.set_center_x(0.5)
        self.text=name
        self.cBox= CheckBox(active=bool(is_here))
        self.add_widget(Label(text=usrid, size_hint_x=0.2) )
        self.add_widget(Label(text=name))
        self.add_widget(self.cBox)

    def return_cBox(self):
        return self.cBox

class AttendanceList(BoxLayout):
    def __init__(self, **kwargs):
        super(AttendanceList, self).__init__(**kwargs)
        self.list_cbox={}
        Clock.schedule_once(self.init_ui, 0)


    def init_ui(self, dt=0):
        self.ScrollView= self.ids['ListAttendanceView']
        temp = self.ids['Header1']

        self.Listview=GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.Listview.bind(minimum_height= self.Listview.setter('height'))
        for ele in find_match.list_id:
            entry = Entry(ele['id'], ele['name'], ele['is_here'])
            self.list_cbox[ele['id']] = entry.return_cBox()
            self.Listview.add_widget(entry)
        self.ScrollView.add_widget(self.Listview)

    def update_cBox(self):
        for ele in find_match.list_id:
            try:
                self.list_cbox[ele['id']].active = ele['is_here']
            except:
                entry = Entry(ele['id'], ele['name'], ele['is_here'])
                self.list_cbox[ele['id']] = entry.return_cBox()
                self.Listview.add_widget(entry)

    def uncheck_all(self):
        for ele in find_match.list_id:
            ele['is_here'] = 0
            self.list_cbox[ele['id']].active = False
        model.save_embed_vector(find_match.list_id)

class qrtestApp(App):
    def build(self):
        Window.clearcolor = (.4,.4,.4,1)
        Window.size = (400, 500)
        '''
        request_permissions([
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE
        ])
        '''
        sm =ScreenManager()
        sm.add_widget(First(name='first_screen'))
        sm.add_widget(Second(name='second_screen'))
        return sm

    def on_stop(self):
        global capture
        if capture:
            capture.release()
            capture = None

qrtestApp().run()