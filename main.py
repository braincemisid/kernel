from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar
from kivy.config import Config
# No resizable window
Config.set('graphics', 'resizable', 0)
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.slider import Slider

import gc

# Brain-CEMISID kernel imports
from kernel_braincemisid import KernelBrainCemisid
from sensory_neural_block import RbfKnowledge



class MyPaintElement(Widget):
    def __init__(self, **kwargs):
        super(MyPaintElement, self).__init__(**kwargs)
        self.active = False

    def on_touch_down(self, touch):
        # Check if touch event is inside widget
        if self.collide_point(touch.x, touch.y):
            self._draw_rectange()

    def on_touch_move(self, touch):
        # Check if touch event is inside widget
        if self.collide_point(touch.x, touch.y):
            # If so, draw rectangle
            self._draw_rectange()

    def _draw_rectange(self):
        self.canvas.clear()
        with self.canvas:
            # lets draw a semi-transparent red square
            Color(1, 1, 1, 1, mode='rgba')
            Rectangle(pos=self.pos, size=(self.width*0.9, self.height*0.9))
        self.active = True

    def clear(self, color):
        self.canvas.clear()
        with self.canvas:
            if color == "black":
                # lets draw a semi-transparent red square
                Color(0, 0.65, 0.65, 1, mode='rgba')
            else:
                Color(0, 0.2, 0.2, 1, mode='rgba')
            Rectangle(pos=self.pos, size=(self.width*0.9, self.height*0.9))
        self.active = False

    def mark(self):
        self._draw_rectange()


class MyPaintWidget(GridLayout):

    CODING_SIZE = 4

    def __init__(self, size, **kwargs):
        super(MyPaintWidget, self).__init__(**kwargs)
        self.cols = size
        for index in range(self.cols * self.cols):
            self.add_widget(MyPaintElement())

    def clear(self):
        index = 0
        #with self.canvas.before:
        #    Color(0, .1, .3, 1)  # green; colors range from 0-1 instead of 0-255
        #    self.rect = Rectangle(size=self.size,
        #                         pos=self.pos)

        for child in self.children:
            if index % 2:
                child.clear("dark-turquoise")
            else:
                child.clear("black")
            index += 1

    def get_pattern(self):
        # Initial pattern is an empty list of integers
        pattern = []
        # Integer representation or first row of pattern (bottom)
        val = 0
        # Counter to obtain integer value from binary row
        count = 1
        # For each MyPaintElement instance, verify if active and
        # add integer value to val depending on its position (count)
        for child in self.children:
            if child.active:
                val += count
            count *= 2
            # If row over, append to pattern array and
            # start encoding new one
            if count == pow(2, MyPaintWidget.CODING_SIZE):
                pattern.append(val)
                val = 0
                count = 1
        return pattern

    def draw_pattern(self, pattern):
        """ Draw given pattern in painter"""
        for index in range(len(pattern)):
            # Get children in groups of four (As codification was made by groups of four)
            child_offset = index*MyPaintWidget.CODING_SIZE
            child_set = self.children[child_offset:child_offset+MyPaintWidget.CODING_SIZE]
            # Convert current number of pattern into binary
            format_str = "{0:0"+str(MyPaintWidget.CODING_SIZE)+"b}"
            bin_pattern_element = format_str.format(pattern[index])
            # Traverse binary, mark or clear corresponding child
            for j in range(len(bin_pattern_element)):
                if bin_pattern_element[MyPaintWidget.CODING_SIZE-1-j] == "1":
                    child_set[j].mark()
                else:
                    if j % 2:
                        child_set[j].clear("dark-turquoise")
                    else:
                        child_set[j].clear("black")

class RbfCardWidget(GridLayout):

    def __init__(self, width, **kwargs):
        super(RbfCardWidget, self).__init__(**kwargs )
        self.rows = 2
        self.spacing = 10
        self.painter = MyPaintWidget(16, size_hint = (1,0.9))
        self.text_label = Label( text = "", size_hint = (1,0.1))
        self.add_widget(self.painter)
        self.add_widget(self.text_label)
        self.size_hint = (None, None)
        self.size = (width, width)

    def show_knowledge(self, knowledge):
        self.painter.draw_pattern(knowledge.get_pattern())
        self.text_label.text = knowledge.get_class()

    def clear(self):
        self.painter.clear()
        self.text_label.text = ""

class MyGroupPaintWidget(GridLayout):

    def __init__(self, **kwargs):
        super(MyGroupPaintWidget, self).__init__(**kwargs)
        self.rows = 1
        self.cards = [RbfCardWidget(100) for count in range(3)]
        for card in self.cards:
            self.add_widget(card)

    def show_rbf_knowledge(self, knowledge_or_vector):
        for card in self.cards:
            card.clear()
        try:
            length = len(knowledge_or_vector)
        except:
            self.cards[0].show_knowledge(knowledge_or_vector)
            return

        if length > 3:
            return
        for index in range(len(knowledge_or_vector)):
            self.cards[index].show_knowledge(knowledge_or_vector[index])

    def clear(self):
        for card in self.cards:
            card.clear()

class SetInternalVariableWidget(GridLayout):

    def __init__(self, img_file, **kwargs):
        super(SetInternalVariableWidget, self).__init__(**kwargs)
        self.rows = 1
        self.img = Image(source=img_file)
        # Improve rendering via OpenGl
        self.img.mipmap = True
        self.img.size_hint_x = None
        self.img.width = 30
        self.slider = Slider(min=0, max=1, value=0.25, size_hint_x=None, width=235)
        self.label = Label(text = str(self.slider.value),font_size='17sp', size_hint_x=None, width=50)
        self.slider.bind(value=self.on_slider_value_change)
        self.add_widget(self.img)
        self.add_widget(self.slider)
        self.add_widget(self.label)


    def on_slider_value_change(self, obj, val):
        self.label.text = str("{0:.2f}".format(val))

    def value(self):
        return self.slider.value

class ShowInternalVariableWidget(GridLayout):

    def __init__(self, img_file = None, **kwargs):
        super(ShowInternalVariableWidget, self).__init__(**kwargs)
        self.spacing = 5
        self.rows = 1
        if not img_file is None:
            self.img = Image(source=img_file)
            # Improve rendering via OpenGl
            self.img.mipmap = True
            self.img.size_hint_x = None
            self.img.width = 30
            self.add_widget(self.img)

        self.progress_bar = ProgressBar(max=1.0, value=0.5, size_hint_x = None, width=235)
        self.label = Label(text = str(self.progress_bar.value),font_size='17sp', size_hint_x=None, width=50)
        self.add_widget(self.progress_bar)
        self.add_widget(self.label)

    def change_value(self, val):
        self.label.text = str("{0:.2f}".format(val))
        self.progress_bar.value = val

class IntentionsInterface(GridLayout):
    def __init__(self, **kwargs):
        super(IntentionsInterface, self).__init__(**kwargs)
        grid_size = 16
        self.kernel = KernelBrainCemisid()
        self.biology_input = SetInternalVariableWidget('icons/biology.png', size_hint_y = 0.33)
        self.culture_input = SetInternalVariableWidget('icons/culture.png', size_hint_y = 0.33)
        self.feelings_input = SetInternalVariableWidget('icons/feelings.png', size_hint_y = 0.33)
        self.internal_state_biology = ShowInternalVariableWidget(size_hint_y = 0.33)
        self.internal_state_culture = ShowInternalVariableWidget(size_hint_y = 0.33)
        self.internal_state_feelings = ShowInternalVariableWidget(size_hint_y = 0.33)
        self.desired_biology_input = SetInternalVariableWidget('icons/biology.png', size_hint_y = 0.33)
        self.desired_culture_input = SetInternalVariableWidget('icons/culture.png', size_hint_y = 0.33)
        self.desired_feelings_input = SetInternalVariableWidget('icons/feelings.png', size_hint_y = 0.33)
        # Main layout number of columns
        self.rows = 2
        self.load_icons()
        self.declare_thinking_panel()
        self.declare_painters(grid_size)
        self.declare_inputs()
        self.declare_buttons()
        self.add_widgets_layouts()
        # Set windows size
        Window.size = (1127, 700)
        # Clear painters when window draw
        self.win_show_uid = Window.fbind('on_draw',self.clear)
        self.win_format_back_uid = Window.fbind('on_draw', self.format_backgrounds)

    def load_icons(self):
        self.img_eye = Image(source='icons/eye.png')
        # Improve rendering via OpenGl
        self.img_eye.mipmap = True
        self.img_ear = Image(source='icons/ear.png')
        # Improve rendering via OpenGl
        self.img_ear.mipmap = True
        self.img_eye.size_hint = (None,None)
        self.img_ear.size_hint = (None,None)
        self.img_eye.width = 60
        self.img_ear.width = 60

    def declare_painters(self, grid_size):
        self.sight_painter = MyPaintWidget(grid_size)
        self.sight_painter.size_hint = (None, None)
        self.sight_painter.size = (200,200)
        self.hearing_painter = MyPaintWidget(grid_size)
        self.hearing_painter.size_hint = (None, None)
        self.hearing_painter.size = (200,200)

    def declare_thinking_panel(self):
        self.thinking_panel = GridLayout(cols=1, size_hint_x=0.6)
        self.thinking_sight = MyGroupPaintWidget(padding=2*self.height/3)
        self.thinking_hearing = MyGroupPaintWidget(padding=2*self.height/3)
        self.thinking_panel.add_widget(self.thinking_sight)
        self.thinking_panel.add_widget(self.thinking_hearing)

    def declare_inputs(self):
        self.hearing_class_input = TextInput(text="Class?")
        self.hearing_class_input.size_hint = (None,None)
        self.hearing_class_input.width = 100
        self.hearing_class_input.height = 35
        self.hearing_class_input.font_size = 18

    def declare_buttons(self):
        # sight clear
        self.sight_clear_btn = Button(text="Clear S", font_size='20sp')
        self.sight_clear_btn.bind(on_press=self.sight_clear)

        # Hearing clear
        self.hearing_clear_btn = Button(text="Clear H", font_size='20sp')
        self.hearing_clear_btn.bind(on_press=self.hearing_clear)

        # Bum btn
        self.bum_btn = Button(text="Bum", font_size='20sp')
        self.bum_btn.bind(on_release=self.bum)

        # Bip btn
        self.bip_btn = Button(text="Bip", font_size='20sp')
        self.bip_btn.bind(on_release=self.bip)

        # Check btn
        self.check_btn = Button(text="Check", font_size='20sp')
        self.check_btn.bind(on_release=self.check)

        # Clack btn
        self.clack_btn = Button(text="Clack", font_size='20sp')
        self.clack_btn.bind(on_release=self.clack)

        # Toggle button (Words, Numbers)
        self.episodes_tgl_btn = ToggleButton(text="Episodes", font_size='20sp', group="bbcc_protocol", state="down", allow_no_selection=False)
        self.intentions_tgl_btn = ToggleButton(text="Intentions",font_size='20sp', group="bbcc_protocol",
                                             allow_no_selection=False)

    def add_widgets_layouts(self):
        # Sight panel
        self.sight_panel = GridLayout(rows=1, padding=10, spacing=10, size_hint_y=0.375)
        self.sight_panel.add_widget(self.img_eye)
        self.sight_panel.add_widget(self.sight_painter)
        # Hearing panel
        self.hearing_painter_text = GridLayout(cols=1)
        self.hearing_painter_text.add_widget(self.hearing_painter)
        self.hearing_painter_text.add_widget(self.hearing_class_input)
        self.hearing_panel = GridLayout(rows=1, padding=10, spacing=10, size_hint_y=0.375)
        self.hearing_panel.add_widget(self.img_ear)
        self.hearing_panel.add_widget(self.hearing_painter_text)

        self.main_panel = GridLayout(cols=2, size_hint=(1,0.9))
        self.senses_bcf_panel = GridLayout(cols=1, padding=25, size_hint_y=0.25)
        self.senses_bcf_panel.add_widget(self.biology_input)
        self.senses_bcf_panel.add_widget(self.culture_input)
        self.senses_bcf_panel.add_widget(self.feelings_input)
        self.senses_panel = GridLayout(cols=1, padding=10, size_hint_x=0.3)
        self.senses_panel.add_widget(self.sight_panel)
        self.senses_panel.add_widget(self.hearing_panel)
        self.senses_panel.add_widget(self.senses_bcf_panel)


        self.main_panel.add_widget(self.senses_panel)


        self.internal_state_panel = GridLayout(cols=1, size_hint_y = 1, size_hint_x = 0.5)
        self.internal_state_label = Label(text="Internal State",font_size='25sp', size_hint_x = None, size_hint_y = 0.3)
        self.internal_state_panel.add_widget(self.internal_state_label)
        self.internal_state_panel.add_widget(self.internal_state_biology)
        self.internal_state_panel.add_widget(self.internal_state_culture)
        self.internal_state_panel.add_widget(self.internal_state_feelings)

        self.desired_state_panel = GridLayout(cols=1, padding_x = 10, size_hint_y = 1,size_hint_x=0.5)
        self.desired_state_label = Label(text="Desired State",font_size='25sp', size_hint_y = 0.3)
        self.desired_state_panel.add_widget(self.desired_state_label)
        self.desired_state_panel.add_widget(self.desired_biology_input)
        self.desired_state_panel.add_widget(self.desired_culture_input)
        self.desired_state_panel.add_widget(self.desired_feelings_input)

        self.desired_internal_states_panel = GridLayout(rows=1, padding=25, size_hint_x=1)
        self.desired_internal_states_panel.add_widget(self.desired_state_panel)
        self.desired_internal_states_panel.add_widget(self.internal_state_panel)

        self.thinking_panel.add_widget(self.desired_internal_states_panel)
        self.main_panel.add_widget(self.thinking_panel)

        # Add widgets to bottom layout
        self.buttons_panel = GridLayout(rows=1, size_hint=(1,0.1))
        self.buttons_panel.add_widget(self.bum_btn)
        self.buttons_panel.add_widget(self.bip_btn)
        self.buttons_panel.add_widget(self.check_btn)
        self.buttons_panel.add_widget(self.clack_btn)
        self.buttons_panel.add_widget(self.episodes_tgl_btn)
        self.buttons_panel.add_widget(self.intentions_tgl_btn)
        self.buttons_panel.add_widget(self.sight_clear_btn)
        self.buttons_panel.add_widget(self.hearing_clear_btn)
        self.add_widget(self.main_panel)
        self.add_widget(self.buttons_panel)

    def learn(self,obj):
        return

    def hearing_clear(self, obj):
        self.hearing_painter.clear()
        self.hearing_class_input.text = "Class?"
        self.thinking_hearing.clear()
        return

    def sight_clear(self, obj):
        self.sight_painter.clear()
        self.thinking_sight.clear()
        return

    def bum(self,obj):
        self.pass_kernel_inputs()
        self.kernel.bum()
        return

    def bip(self,obj):
        self.pass_kernel_inputs()
        self.kernel.bip()
        self.show_kernel_outputs()
        return

    def check(self,obj):
        self.pass_kernel_inputs()
        self.kernel.check()
        self.show_kernel_outputs()
        return


    def clack(self,obj):
        self.pass_kernel_inputs()
        self.kernel.clack()
        self.show_kernel_outputs()
        return

    def pass_kernel_inputs(self):
        # Set working domain
        if self.episodes_tgl_btn.state == "down":
            self.kernel.set_working_domain("EPISODES")
        else:
            self.kernel.set_working_domain("INTENTIONS")
        # Get patterns
        hearing_pattern = self.hearing_painter.get_pattern()
        sight_pattern = self.sight_painter.get_pattern()
        hearing_class = self.hearing_class_input.text
        hearing_knowledge = RbfKnowledge(hearing_pattern, hearing_class)
        sight_knowledge = RbfKnowledge(sight_pattern, "NoClass")
        self.kernel.set_hearing_knowledge_in(hearing_knowledge)
        self.kernel.set_sight_knowledge_in(sight_knowledge)
        biology_in = self.biology_input.value()
        culture_in = self.culture_input.value()
        feelings_in=self.feelings_input.value()
        self.kernel.set_internal_state_in([biology_in, culture_in, feelings_in])

    def show_kernel_outputs(self):
        self.thinking_clear()
        self.hearing_class_input.text = self.kernel.state
        kernel_internal_state = self.kernel.get_internal_state().get_state()
        self.internal_state_biology.change_value(kernel_internal_state[0])
        self.internal_state_culture.change_value(kernel_internal_state[1])
        self.internal_state_feelings.change_value(kernel_internal_state[2])
        if self.kernel.state == "HIT":
            h_knowledge = self.kernel.get_hearing_knowledge_out()
            s_knowledge = self.kernel.get_sight_knowledge_out()
            if h_knowledge is not None:
                self.thinking_hearing.show_rbf_knowledge(h_knowledge)
            if s_knowledge is not None:
                self.thinking_sight.show_rbf_knowledge(s_knowledge)

    def thinking_clear(self):
        self.thinking_sight.clear()
        self.thinking_hearing.clear()

    def clear(self, obj):
        self.sight_clear(None)
        self.hearing_clear(None)
        self.thinking_clear()
        Window.unbind_uid('on_draw', self.win_show_uid)

    def set_zero(self, obj):
        self.pass_kernel_inputs()
        self.kernel.set_zero()
        return

    def set_add_operator(self, obj):
        self.pass_kernel_inputs()
        self.kernel.set_add_operator()
        return

    def set_equal_sign(self, obj):
        self.pass_kernel_inputs()
        self.kernel.set_equal_sign()
        return

    def format_backgrounds(self, obj):
        with self.senses_panel.canvas.before:
            Color(0.11, 0.10, 0.1, 1, mode='rgba')
            Rectangle(pos=self.senses_panel.pos, size=(self.senses_panel.width, self.senses_panel.height))
        with self.desired_internal_states_panel.canvas.before:
            Color(0.07, 0.07, 0.07, 1, mode='rgba')
            Rectangle(pos=self.desired_internal_states_panel.pos, size=(self.desired_internal_states_panel.width, self.desired_internal_states_panel.height))
        Window.unbind_uid('on_draw', self.win_format_back_uid)

class MyPaintApp(App):
    def build(self):
        intentions_ui = IntentionsInterface()
        return intentions_ui

if __name__ == '__main__':
    MyPaintApp().run()
