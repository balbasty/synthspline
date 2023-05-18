import tkinter as tk
from tkinter import *
import json
import os

class VesselsynthGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vesselsynth")
        self.root.geometry("400x300")
        self.root.configure(bg="black")
        self.button_homescreen_text_size = 30
        self.json_window_font = ("Arial", 30, "bold")
        self.json_entry_font = ("Arial", 20)

        self.create_buttons()

    def create_buttons(self):
        button1 = tk.Button(self.root, text="Synthesize Vessels", command=self.vesselsynthJson, bg="#bba6aa", fg="black", font=("Arial", self.button_homescreen_text_size, "bold"), padx=20, pady=10).pack(pady=10)
        button2 = tk.Button(self.root, text="View OCT Images", command=self.list_vessels, bg="#bba6aa", fg="black", font=("Arial", self.button_homescreen_text_size, "bold"), padx=20, pady=10).pack(pady=10)
        button3 = tk.Button(self.root, text="Train U-Net", command=lambda: self.button_click(3), bg="#bba6aa", fg="black", font=("Arial", self.button_homescreen_text_size, "bold"), padx=20, pady=10).pack(pady=10)

        self.root.config(padx=20, pady=20)

    def vesselsynthJson(self):
        json_window = tk.Toplevel(self.root, padx=50)
        json_window.title("vesselsynth_params.json")
        self.readData(json_window, "vesselsynth_params.json")

    def readData(self, window, filename):
        
        with open(filename) as file:
            json_data = json.load(file)

            entries = {}
            i = 0
            for key in json_data:
                tk.Label(window, text=key, font=self.json_window_font).grid(row=i,column=0, sticky=W)
                entry = tk.StringVar(window, value=json_data[key])
                tk.Entry(window, width=50, bd=5, textvariable=entry, font=self.json_entry_font).grid(row=i,column=1, sticky=tk.E)
                entries[key] = entry
                tk.Button(window, text="info").grid(row=i, column=2)
                i+=1

            def saveJson():
                for key in json_data:
                    json_data[key] = entries[key].get()

                print("saving new json file:", json_data)

            save_button = tk.Button(window, text="Save", command=saveJson, bg="#3498db", fg="black", font=("Arial", 12), padx=10, pady=5)
            save_button.grid(pady=10, row=i, column=1)


    def open_json_window(self):
        name_var='hey'
        json_window = tk.Toplevel(self.root)
        json_window.title("vesselsynth_params.json")
        lab = tk.Label(json_window, text="First Name").pack()
        #tk.Entry(json_window, bd=5, textvariable=name_var).insert(END, 'thingy')#.pack()

        name = StringVar(json_window, value='not available')
        nameTf = Entry(json_window, textvariable=name).pack()


        #with open("vesselsynth_params.json") as file:
        #    json_data = json.load(file)

        #text_widget = tk.Text(json_window, height=20, width=50, font=("Arial", 40), bg="#f2f2f2", fg="#333333", padx=10, pady=10)
        #text_widget.pack()

        #text_widget.insert(tk.END, json.dumps(json_data, indent=4))

        #def save_json():
        #    updated_json = text_widget.get("1.0", tk.END)
        #    updated_data = json.loads(updated_json)
        #    with open("data.json", "w") as file:
        #        json.dump(updated_data, file, indent=4)
        #    
        #    json_window.destroy()

        #save_button = tk.Button(json_window, text="Save", command=save_json, bg="#3498db", fg="black", font=("Arial", 12), padx=10, pady=5)
        #save_button.pack(pady=10)

    def list_vessels(self):
        vessels_window = tk.Toplevel(self.root)
        vessels_window.title("Vessels")
        vessels_window.geometry("300x300")

        folder_path = "folder"  # Update with your folder path

        files = os.listdir(folder_path)

        for file in files:
            button = tk.Button(vessels_window, text=file, command=lambda file=file: self.view_vessel(file), bg="#1abc9c", fg="black", font=("Arial", 12, "bold"), padx=10, pady=5)
            button.pack(pady=5)

    def view_vessel(self, file):
        # Placeholder function to demonstrate opening a vessel file
        print(f"Viewing vessel: {file}")

    def button_click(self, button_number):
        if button_number == 1:
            print("Button 1 clicked!")
       
    def button_click(self, button_number):
        if button_number == 1:
            print("Button 1 clicked!")
        elif button_number == 2:
            print("Button 2 clicked!")
        elif button_number == 3:
            print("Button 3 clicked!")

    def run(self):
        self.root.mainloop()

# Create an instance of the VesselsynthGUI class and run the GUI
gui = VesselsynthGUI()
gui.run()
