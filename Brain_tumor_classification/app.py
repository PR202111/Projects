import gradio as gr
import pickle
import numpy as np
from PIL import Image

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def recognize_img(image):
    img = Image.fromarray(image).resize((128,128))

    x = np.array(img)
    x = x.reshape(1,128,128,3)

    res = model.predict_on_batch(x)
    res = np.argmax(res[0])
    classification = ['Tumor','No, Tumor']
    return f'{classification[res]}' 

examples = []
examples.append('No22.jpg')
examples.append('Y257.jpg')

# #@title You can also add some description and explanation
# #  to your app's interace if you want. Go ahead and specify some text for the description and the long description (if you want to):
desc = "Brain tumor app. Let's learn!" # @param {type:"string"}
long_desc = "Select an image or upload one to predict if brain tumor is present or not" # @param {type:"string"}
heading_title = 'Project 1 BRAIN TUMOUR DETECTION'

theme_selection = 'Glass'

theme_dict = {
    "Base": gr.themes.Base(),
    "Default": gr.themes.Default(),
    "Glass": gr.themes.Glass(),
    "Monochrome": gr.themes.Monochrome(),
    "Soft": gr.themes.Soft()
}

selected_theme = theme_dict[theme_selection]


# Assuming recognize_image, examples, heading_title, desc, long_desc, and selected_theme are defined elsewhere.

# Update the import for components
image = gr.Image()
label = gr.Label()

# Create the interface with the updated component imports
iface = gr.Interface(
    fn=recognize_img,
    inputs=image,
    outputs=label,
    examples=examples,
    title=heading_title,
    description=desc,
    article=long_desc,
    theme=selected_theme  # Make sure this is defined based on user selection as explained in previous messages
)

iface.launch(share=True, debug=True)

