import json
import tkinter as tk


class GuiFiles(object):

    def __init__(self):
        self.json_data = json.load(open("vesselsynth/vesselsynth_params.json"))
        self.keys = self.json_data.keys()

    def newWindow(self, title):
        new_window = tk.Toplevel(self.root, padx=50)
        new_window.title(title)
        return new_window
    
    def dataWindow(self):
        print("pretend the window just popped up")
        for key in self.keys:
            print(f"value for {key}?")
            self.json_data[key] = input('> ')
        print(self.json_data)

    def saveData(self):
        ##save_button = tk.Button(window)


GuiFiles().dataWindow()