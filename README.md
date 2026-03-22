Hi, I'm Guido, I'm 12 and this is my LLM project.
This is Eisi, an open-source AI designed to help you organize your appointments.
It's still in alpha, so performance is limited and not guaranteed to work in every situation.

🗣️ Eisi can now speak English!


⚠️ Important Notes

The dataset is focused only on organizing appointments.
General-purpose use is not recommended yet.
The model is incomplete and still experimental.
This project is open-source.
To run Eisi correctly, you must download all files or clone the repository. Missing files will cause errors.
Eisi must be run on Python 3.13.0. Other versions are not supported and may cause unexpected behavior.


🔧 Customization
You can replace the dataset with your own if you want.
Everything here is free to use, modify, and experiment with.
This is just the beginning — the final version will be much more complete.

🛠️ How to Install Eisi
Clone the repository from your terminal:

```
git clone https://github.com/tuo-username/Eisi.git
```

Enter the project folder:

```
cd Eisi
```

Make sure you have Python 3.13.0 installed, then install all required libraries as listed below.

⚠️ Testing is recommended only for users comfortable with Python environments and debugging.


📦 Library Installation Instructions
When you download the project, open it in an editor such as Visual Studio Code.


---

✅ Modules already included in Python (no installation required)

These libraries come bundled with Python:

re  
dataclasses (included by default in Python 3.7+)  
collections (Counter, defaultdict)  
math  

---

📥 Libraries that must be installed manually

Install these packages using pip:

torch  
pythorc (if this is a custom module; otherwise PyTorch is simply torch)  
matplotlib  
venv

---

🧩 How to install them

🟦 On Windows

Use pip:

```
pip install torch  
pip install matplotlib    
pip install torch  
pip install pytorch
```

Then create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate
``

🍏 On macOS

Use pip3, since this project requires Python 3 only:

```
pip3 install torch  
pip3 install matplotlib  
pip3 install torch  
pip3 install pytorch  
python3 -m venv venv
```

Then create and activate a virtual environment:

```
python -m venv venv
```

```
source venv/bin/activate
```

🐧 On Linux

Use pip3, since this project requires Python 3 only:

```
pip3 install torch  
pip3 install matplotlib  
pip3 install torch  
pip3 install pytorch  
python3 -m venv venv
```

Then create and activate a virtual environment:

```
python -m venv venv
```

```
source venv/bin/activate
```


🧨 Final Notes

Install all modules and libraries listed above without exceptions.
Run every installation step from your terminal or the integrated terminal in your code editor (e.g. Visual Studio Code).
This project is in early alpha with a limited dataset and experimental behavior — recommended only for users comfortable with errors, dependencies, and manual setup.
If you are an advanced user, feel free to test, modify, and improve the project — every contribution helps Eisi grow.


📄 License
This project is licensed under the MIT License. You are free to use, modify, distribute, and experiment with the code under its terms.
