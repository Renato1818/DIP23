import tkinter as tk
from tkinter import Menu

### FUNCTIONS ###

# Function to close the window
def close_window():
    window.destroy()

# Create a button callback function
def button_click():
    label.config(text="Button Clicked!")    

# Function to open a new window
def open_new_window():
    new_window = tk.Toplevel(window)
    new_window.title("New Window")
    new_label = tk.Label(new_window, text="This is a new window")
    new_label.pack()

###   CODE   ###

# Create a main window
window = tk.Tk()
window.title("Experimental GUI")

# Set the dimensions (width x height)
window.geometry("400x200")

# Create a menu bar
menu_bar = tk.Menu(window)
window.config(menu=menu_bar)

# Create a "File" menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)

# Add options to the "File" menu
file_menu.add_command(label="New Window", command=open_new_window)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=close_window)

# Create a label
label = tk.Label(window, text="Hello, GUI!")
label.pack()

# Create a button callback function
def button_click():
    label.config(text="Button Clicked!")

# Create a button
button = tk.Button(window, text="Click me", command=button_click)
button.pack()

# Start the GUI event loop
window.mainloop()
