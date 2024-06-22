# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from PIL import Image
import base64
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

# Custom deserialization function for BatchNormalization
def custom_batchnorm_layer(config):
    if 'axis' in config:
        config['axis'] = config['axis'][0] if isinstance(config['axis'], list) else config['axis']
    return BatchNormalization(**config)

# Custom objects dictionary to handle the custom deserialization function
custom_objects = {
    'BatchNormalization': custom_batchnorm_layer,
}

# File path to your saved Keras model
model_path = "C:\\Users\\HP\\OneDrive\\Documents\\Medicinal-Leaf-Classification-main\\artifacts\\updatednewmodel.h5"

# Load the Keras model with custom objects
try:
    model = load_model(model_path, custom_objects=custom_objects)
    st.write("Keras model loaded successfully.")
except OSError as e:
    st.write(f"Error loading Keras model: {e}")
except Exception as e:
    st.write(f"Unexpected error loading Keras model: {e}")

# File path to your NumPy array
class_dict_path = "C:\\Users\\HP\\OneDrive\\Documents\\Medicinal-Leaf-Classification-main\\artifacts\\updatedclassnames.npy"

# Load the NumPy array
try:
    class_dict = np.load(class_dict_path, allow_pickle=True).item()
    st.write("NumPy array loaded successfully.")
except FileNotFoundError:
    st.write(f"Error: File '{class_dict_path}' not found.")
except Exception as e:
    st.write(f"Error loading NumPy array: {e}")

#model = load_model("C:\\Medicinal-Leaf-Classification-main\\Notebooks\\Resnetmodel.h5")
#class_dict = np.load("C:\\Medicinal-Leaf-Classification-main\\Notebooks\\newclass_names.npy", allow_pickle=True)

def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 3


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="",  # required
                options=["Home", "Predict", "Contact"],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
         with st.sidebar:
            selected = option_menu(
                menu_title="𝓐𝔂𝓾𝓻𝓢𝓬𝓪𝓷",  # required
                options=["Home", "Predict", "Remedies"],  # required
                icons=["house", "upload", "book"],  # optional
                menu_icon="🌳",  # optional
                default_index=0,  # optional
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "15px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                },
                    "nav-link-selected": {"background-color": "green"},
            },
        )
            return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":
    #st.title(f"You have selected {selected}")
    #st.markdown('<h1 style="color: green;">Identification of Medicinal plant using Machine learning</h1>', unsafe_allow_html=True)
    def add_bg_from_local(image_path):
        image = Image.open(image_path)
        st.markdown(
            f'<style>body{{background-image: url("data:image/png;base64,{image_to_base64(image)}");background-size: cover;}}</style>', 
            unsafe_allow_html=True
    )

# Function to convert image to base64 format
    def image_to_base64(image):
        from io import BytesIO
        import base64

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

# Function to display the home page with specified medicinal plants information and images
    def home_page():
    #st.title("Medicinal Plants Identification")MediBotanical
    #st.markdown('<h1 style="color: green;">MediBotanical</h1>', unsafe_allow_html=True)
        st.markdown(
        '<h1 style="color: green;"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpg6wbC1dGFZaQwT8VoFq4FnU6UqmsbPD-IQW3pKZOAU5hd-LfUQM8EStFoye0UIhW8os&usqp=CAU" style="width: 100px;">𝓐𝔂𝓾𝓻𝓢𝓬𝓪𝓷 </h1>', 
        unsafe_allow_html=True
    )
        st.markdown('<h1 style="color: green;">Medicinal Plants Identification</h1>', unsafe_allow_html=True)

    #st.header("Welcome to the Home Page!")
        st.markdown('<h2 style="color: green;">Welcome to the Home Page!</h2>', unsafe_allow_html=True)
        st.write("We have collected data for 40 plant species related to medicinal use. Here are some samples:")
        st.write("Medicinal plants have been used in traditional medicine practices for a long time because of their nutrients and medicinal properties.")
        st.write("Due to their bioactive compounds, such as phenolic, carotenoid, anthocyanin, and other bio-active components, they are known for their antioxidant, anti-allergic, anti-inflammatory, and antibacterial properties. Different species of plants are recognized as having medical properties; they can be trees, shrubs or herbs.")
    #image_path = r"C:\Users\HP\OneDrive\Documents\Medicinal-Leaf-Classification-main\plant.png"
        image_url = "https://www.smallsteps.org.nz/images/heros/tools/identifying-signals-desktop.png"
        st.image(image_url, use_column_width=True)
    # Medicinal plant information
        st.write("Here, Some of the example of medicnal plant")
        medicinal_plants = [
            {
                "name": "Piper betle (Betel)",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Betel/1880.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T144845Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=497b4d1b7d51052f6c381773e7a6d63d24fe1f727bb58274e499c105103ea683e51702c6590229478a62a1e384f1ad1fc4836074504a389dd35e0ff11f4ddf455a76211e64bccda6fdede8d875716c9b7f7f05b25c6f6c4dd43ef63191a7ef54ab5560e3fabb83dda226f74a33e1b6a3ccc8363aaf659933354a3c5b4032a47f677a604c38d10d8400afb31855a0ec2df79b642a83878c385096962f84084c5d8ab1af2c5052110b321b28f3a0939a4f54cb426553d0263ff2a138494c71efed5b40e3cbf6f1e3ef5dd259ef44adc3d60cab83733b7999b0d6e7cc0c3d547c6dcb5f95250f8bdb2b72b054cada56f761f6d052461551c1fadfd7238ad3b9d1e8",
                "description": "Piper betle, also known as Betel, is a vine belonging to the Piperaceae family. Its leaves are commonly chewed for their stimulant properties."
            },
            {
                "name": "Murraya koenigii (Curry)",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Curry/390.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T144928Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=1e781654ef8f517d76495905b83343fe88a194fb6364ce2b942a0185516ec439d0a99b56cbcb0a8383120f20a4fcf5350a752d5cddfdb90610132408979eaeff4bb216e9f6514966204acc1df9c4a4ddd38b7ac299d6375947563380978b84d0bbacffeddab9d7365b4314b697cf5d1bd1f714ca000ac08a5cb96e3a6ac43faa30d20d43296b559e3bc2781cdaa7148c1484ff77f004d0982cbf9d9526d981c3f88a75dc1b5feb0a52b111319125c8da338478785b02164121d4ad139e73021d4e5142e697b50ced2abebde7fbc6c2478252e28b23a923aa2dc792b61ae9b14efe0bdc3262347432d51088e69be10c270cd6b6dcc8aa556aa65f54189391f883",
                "description": "Murraya koenigii, commonly known as Curry or Curry Leaf, is a tropical tree native to India. Its leaves are used as a culinary ingredient."
            },
            {
                "name": "Guava (leaf)",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Guava/176.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145000Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=1056823aef498dd7c39ecda132dce2e49d6a7737918a0da177d4cca951b092e4808ffede3af9954282f56dd8b819ebea043baf5a4b68f1508d8ac25dce97b7e64e872de0e1b4b7ef7ec1a4308b8ec4abdbb396198c1627b15f65dcd1a0c1f36b066a426fb07f72bec7acdcc6c21e1eb0a79ae8d66c91a857d1c002400a1a7bb38f9d3228c92a3b0c1ad58b55bf99d931a21d53a8060a9ab07d526913b05bef8bf30fbe05b2f4601b05729b4b5f55a91055e6f1318ac54f7976eab460d461264ecbc5516f41561cb07cafd38372d2fcaed3ad847f93f9bdd612ab26fe56ed6a5d0a05a80bccb3d72f87cdcb642c7b6f6513b906e46b308d2a91a7ed6327d3b2f8",
                "description": "Guava is a tropical fruit tree with leaves that have medicinal properties. Guava leaves are used in herbal teas and traditional medicine."
            },
            {
                "name": "Hibiscus (leaf)",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Hibiscus/502.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145032Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7babdfa4fa3e4f33d2d3b506634fbc3a80958ed74c35b74ab96b6e72f3b8fce35781c307830f0c3d6bc7aa4c61ad7b33daa01876df676e71d6a3cd0509ce9aae73ccd028fd90b3fc85baa8eda4276a5fdc33f5cd4b0e0084e01d91f0f2f8e368ebba0d6f37a160017513dc26667f43fceb6f428f883ff8c0a5a7b6b8963ff8a70a161880b6df6eb33add5ff4352be2f813306ca389d69fb429ee94f21680ebce93a204a3925717c76136d939fada6731a7c542eec2923382cbd1eeb617f5306101dcd0a6b7069ddccff42743e9c896cd4b59f5c47743734feedf598418862db28626df60730a12a4127dd9cc6359b525bf64ddb266b1e13cdfbd1b02e4d4d048",
                "description": "Hibiscus is a flowering plant with leaves that are used for making herbal teas and extracts. It is known for its potential health benefits."
            },
            {
                "name": "Indian Mustard (leaf)",
                "image_url": "https://m.media-amazon.com/images/I/61kYL0rdC8S._AC_UF1000,1000_QL80_.jpg",
                "description": "Indian Mustard is a plant with nutritious leaves commonly used in Indian cuisine. The leaves are rich in vitamins and minerals."
            },
            {
                "name": "Jackfruit (leaf)",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Jackfruit/184.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145111Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=76d6f76c183cc4d7f4a24d7e10e673b089308a65cf0cf5790537ba6a9f6406b51b4ab9aca211dc6572a84ea0d1fa8a6bde59a8c9a47007d54d57fee2c5102cb443c453656f6dfe08b1155cef8edd3e22a761ed2e3ae1e3f262e08f5e87a928cbde92486fa0b3021a456f7b0dde3f62f9816853915b854ec52786df686ca283b25ff431feb7df267b7a2270a98ae5596cc369d43f1fe07bceba55c5516b033b40675ad4b359b3afd8ea71125e9634c2630b56f81b3aa1c98f111ebe870a1be1d0eea04f22f8006d68da828820fc46685194efe09131b47eefb9d475828fafe1ce1e5845403f874384d2ad0ede8a7316fcdb8a9505e1b9085017057c6865e3f18a",
                "description": "Jackfruit is a tropical fruit tree with large leaves. The leaves have various culinary and medicinal uses in different cultures."
            },
            {
                "name": "Jasmine (leaf)",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Jasmine/436.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145141Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=ca008ef30dbb565bc1a927470d08187923bf90835ae7a51adc70b56db2ea951fd457dc4092f9608895d929eb4b00c4f84b311bd6c5764d7f64f17d39fe93920886d9a902c1c4c50023a084899306f79ae32789e5b59656772399e80348b7a0e58865f52f1b6df17f05985577db6de750d22aba7878bea36c68af9b05d3b153e2b712aa58bf53a4694773012c29ed9b2047781a8c48bcce8179ef0f3a0e1ec1768ab8d8f41a4ddb23a8ef8f3ca7d4de0ff05512aa86a5c5c650b9725a5a8a082ee41b37ebceb3b6fe7300d859be6248f9d13a8c209e91f612424615a0dafabaa9fbc3f832aec3f7a4947fce6b64a074d8903c54b1e807f8329477974a413b6389",
                "description": "Mint is a fragrant herb with leaves commonly used in cooking, beverages, and herbal remedies. Mint leaves have a refreshing flavor."
            },
            {
                "name": "Neem",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Neem/1008.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145211Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=85384e8414d5302bdb3eed41b463b4063a098a39e3885441562ebcbbc57ba40706c29414a97bcc642cc8f4b27611eb6de388a2f43c83f3f598e3cf5758823075d07e420412d8320a7deb7ff8c0fa613ac223440e6fd71853eb5809251afa439ebe80420e125c618f9da97bb9f7b7d0ae61f3e9870f7c9f3ebbffe7d540a0964a29faac435ccab229ecc33d9aae44ed1d0e4cfe69ebfab02312a563b336845e60c42fffb7881211fd4e03141dcb61cf5d9dfa429b748a7e270f07024e290aca6f80290ca5086b23cda6a8185d1588240b2da6651e7ccf530e161e3b0773be55bf60558df4850ba244eef6e6ea78f3785814e1f449ac2ee6487624204bd4cc86ce",
                "description": "Neem, also known as Azadirachta indica, is a tree with medicinal properties. Neem leaves, bark, and oil are used in traditional medicine"
            },
            {
                "name": "Bhringraj",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Bringaraja/IMG_20190909_134521.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145332Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=d07cd7b6fffee629638d0ac68aab8aa64c0339abd4d0d4c745a248ee4820a39254d664194db20527934f4053c16f5fe6e6da71da044970c80df1843ec8ba9c4b1aebde816a25827a46eb3e28d01cfdd595c66b7727c14096ddbc4cd98eaf7de17f661562b9386a5915c1bd15f663d3a35a1dafa2a9cd6866c2015bf325215f6c9ae5bd8879236396310ba698b5600621e3f974d731513122249079cfc8748a3260b281f42e36178ed38548e610f58734940afe14fad8f2711b4748abc0a766d964e07a2dbcaa2dde52b0dd5868807cd167db2b8ed4884d9301c9a22b2c4f824dfc5c1d7b20b7b931c709f1f8f4afd24df8e3bac9052408449f1256c0ac6b56ec",
                "description": "Eclipta prostrata, commonly known as Karisalankanni, and bhringraj, is a species of plant in the family Asteraceae."
            },
            {
                "name": "Nilavembu",
                "image_url": "https://storage.googleapis.com/kagglesdsdata/datasets/3701557/6417582/Indian%20Medicinal%20Leaves%20Image%20Datasets/Medicinal%20Leaf%20dataset/Nelavembu/1180.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240501%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240501T145251Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=02f62e45b7526958435ed0896f3117b48f8b91b33759a4cf3c17303b937703251e36030b170be2771f553378e30f076d76cd1e3f554e3d05bca68a807f2ac0e72992a78e389db671c42190825d9caf278ce01db53c0055f9ef51e4798a79befd5b0df2dd7d12a2958a38dfef1e0a7839bb4f210b536ca18347c32166ff532fd3e4b5f23f0010de9a54fd2eecb72a20a37ee65335ed625de35306ceebead69be4a98ee67aa4b2f8ebf133578272714c4d8898b6033d79e64f7bebd67c0312a2053d6534729905297f2550de1daf7d5bb9efdc405ac8ce7cf9ac3d47918f1442926d247167f944175eacfc77abaf4de040bade1d78ab7ff9398e8356b01d1a01d0",
                "description": "Andrographis paniculata, commonly known as creat or green chiretta, is an annual herbaceous plant in the family Acanthaceae. "
            }
            
        ]
        background_color = "#AFE1AF"
        st.markdown(
            f"""
            <style>
                .st-eb {{
                background-color: {background_color};
                padding: 10px;
                border-radius: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Wrap the columns inside an expander
        with st.expander("Medicinal Plants"):
            col1, col2 = st.columns(2)

        for i, plant in enumerate(medicinal_plants):
            with col1 if i % 2 == 0 else col2:
                st.subheader(plant["name"])
                st.image(plant["image_url"], use_column_width=True)
                st.write(plant["description"])

    #for plant in medicinal_plants:
    #    st.subheader(plant["name"])
    #    st.image(plant["image_url"], caption=plant["description"], width=300)
    # Display medicinal plants in two columns
        #col1, col2 = st.columns(2)

        #for i, plant in enumerate(medicinal_plants):
            #with col1 if i % 2 == 0 else col2:
                #st.subheader(plant["name"])
                #st.image(plant["image_url"], use_column_width=True)
                #st.write(plant["description"])

# Main function to run the Streamlit app
    def main():
    # Add background image
    #add_bg_from_local("Background.jpg")
        home_page()

# Run the main function
    if __name__ == "__main__":
        main()

if selected == "Predict":
    #st.title(f"Welcome to ")
    translator = Translator()
    st.markdown(
        '<h1 style="color: green;"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpg6wbC1dGFZaQwT8VoFq4FnU6UqmsbPD-IQW3pKZOAU5hd-LfUQM8EStFoye0UIhW8os&usqp=CAU" style="width: 100px;">𝓐𝔂𝓾𝓻𝓢𝓬𝓪𝓷 </h1>', 
        unsafe_allow_html=True
    )

    #st.markdown('<h1 style="color: green;">Welcome to MediBotanical</h1>', unsafe_allow_html=True)
    #def get_uses(class_name):
    # Assuming you have a dictionary mapping class names to their uses or information
    uses_dict = {
                'Azadirachta Indica (Neem)': {
                    'Use 1': 'Neem leaf is used for leprosy, eye disorders, bloody nose, intestinal worms, stomach upset, loss of appetite, skin ulcers, diseases of the heart and blood vessels (cardiovascular disease), fever, diabetes, gum disease (gingivitis), and liver problems. The leaf is also used for birth control and to cause abortions, The bark is used for malaria, stomach and intestinal ulcers, skin diseases, pain, and fever. The flower is used for reducing bile, controlling phlegm, and treating intestinal worms.',
                    'Use 2': '(வேம்பு) தொழுநோய், கண் கோளாறுகள், இரத்தம் தோய்ந்த மூக்கு, குடல் புழுக்கள், வயிற்று வலி, பசியின்மை, தோல் புண்கள், இதயம் மற்றும் இரத்த நாளங்களின் நோய்கள் (இருதய நோய்), காய்ச்சல், நீரிழிவு, ஈறு நோய் (ஈறு அழற்சி) மற்றும் கல்லீரல் ஆகியவற்றிற்கு வேப்ப இலை பயன்படுத்தப்படுகிறது. பிரச்சனைகள். இந்த இலை பிறப்பு கட்டுப்பாடு மற்றும் கருக்கலைப்புக்கு பயன்படுத்தப்படுகிறது, பட்டை மலேரியா, வயிறு மற்றும் குடல் புண்கள், தோல் நோய்கள், வலி ​​மற்றும் காய்ச்சல் ஆகியவற்றிற்கு பயன்படுத்தப்படுகிறது. பித்தத்தைக் குறைக்கவும், சளியைக் கட்டுப்படுத்தவும், குடல் புழுக்களைக் குணப்படுத்தவும் பூ பயன்படுகிறது.'
                    },            
                'Amaranthus Viridis (Arive-Dantu)': {
                    'use 1':'The leaves are used to treat fever. The poultice prepared from leaves is used to treat inflammation, boils and abscess. It is also used as a remedy for lung disorders.',
                    'use 2':'(தண்டுக்கீரை) இலைகள் காய்ச்சலுக்குப் பயன்படுத்தப்படுகின்றன. இலைகளில் இருந்து தயாரிக்கப்படும் பூல்டிஸ் வீக்கம், கொதிப்பு மற்றும் சீழ் போன்றவற்றை குணப்படுத்த பயன்படுகிறது. நுரையீரல் கோளாறுகளுக்கு மருந்தாகவும் பயன்படுகிறது.'
                    },
                'Artocarpus Heterophyllus (Jackfruit)':{
                    'use 1':'Jackfruit contains plenty of iron. Consumption of Jackfruit might help maintain the health of the thyroid gland. Jackfruit contains a high amount of copper, jackfruit leaves can be cooked as a plain dry dish and in some parts of the country, people cook them along with boiled green grams',
                    'use 2':'(பலாப்பழம்) பலாப்பழத்தில் இரும்புச்சத்து அதிகம் உள்ளது. பலாப்பழத்தை உட்கொள்வது தைராய்டு சுரப்பியின் ஆரோக்கியத்தை பராமரிக்க உதவும். பலாப்பழத்தில் அதிக அளவு தாமிரம் உள்ளது, பலாப்பழ இலைகளை வெற்று உலர் உணவாக சமைக்கலாம் மற்றும் நாட்டின் சில பகுதிகளில், மக்கள் அவற்றை வேகவைத்த பச்சைப்பயறுகளுடன் சேர்த்து சமைக்கிறார்கள்.'
                    },
                'Basella Alba (Basale)': {
                    'use 1':'Basale Controls gestational diabetics, treatment of mouth ulcer, It is the main ingredient of the Ayurvedic tonic Sukhaprasava gritham which advices to consume from 8th month of pregnancy, Leaves and stems are used for treating  Anemia, Cancer in stomach and osteoporosis',
                    'use 2':'(பசலை கீரை) ஆயுர்வேத டானிக் சுகப்ரசவ கிரித்தத்தின் முக்கிய மூலப்பொருளாகும், இது கர்ப்பத்தின் 8 வது மாதத்திலிருந்து உட்கொள்ள அறிவுறுத்துகிறது, இலைகள் மற்றும் தண்டுகள் இரத்த சோகை, வயிற்றில் புற்றுநோய் மற்றும் எலும்புப்புரை சிகிச்சைக்கு பயன்படுத்தப்படுகின்றன.'
                    },
                'Brassica Juncea (Indian Mustard)':{
                    'use 1':'Indian Mustard is a dietary supplement made from the seeds of the Brassica juncea plant. It is rich in vitamins, minerals, and antioxidants, and is believed to have many health benefits, including aiding digestion, reducing inflammation, and helping to lower cholesterol.',
                    'use 2':'(கடுகு) வைட்டமின்கள், தாதுக்கள் மற்றும் ஆக்ஸிஜனேற்றங்கள் நிறைந்துள்ளன, மேலும் செரிமானத்திற்கு உதவுதல், வீக்கத்தைக் குறைத்தல் மற்றும் கொழுப்பைக் குறைக்க உதவுதல் உள்ளிட்ட பல ஆரோக்கிய நன்மைகள் இருப்பதாக நம்பப்படுகிறது.'
                    },     
                'Carissa Carandas (Karanda)': {
                    'use 1':'Various parts (fruits, leaves, bark and roots) of Carissa carandas are popular for their medicinal use in diarrhea, constipation, malaria, epilepsy, neurological disorder, pain, myopathic spams, leprosy, anorexia, cough, pharangitis, diabetes, seizures, scabies and fever',
                    'use 2':'(களாக்காய்) வயிற்றுப்போக்கு, மலச்சிக்கல், மலேரியா, கால்-கை வலி, நரம்பியல் கோளாறு, காய்ச்சல், தசைப்பிடிப்பு, தொழுநோய், பசியின்மை, இருமல், பாராங்கிடிஸ், நீரிழிவு, வலிப்பு, சிரங்கு போன்றவற்றில் கரிசா காரண்டாஸின் பல்வேறு பாகங்கள் (பழங்கள், இலைகள், பட்டை மற்றும் வேர்கள்) மருத்துவப் பயன்பாட்டிற்காக பிரபலமாக உள்ளன.'
                    },
                'Citrus Limon (Lemon)':{
                    'use 1': 'Lemon contain Vitamin C , high blood pressure, the common cold, and irregular menstruation. limon is a known remedy for coughs',
                    'use 2':'(எலுமிச்சை) வைட்டமின் சி, எலுமிச்சை உயர் இரத்த அழுத்தம், ஜலதோஷம் மற்றும் ஒழுங்கற்ற மாதவிடாய்க்கு உதவுகிறது. எலுமிச்சை இருமலுக்கு மருந்தாகும்'
                    },
                'Ficus Auriculata (Roxburgh fig)': {
                    'use 1':'Roxburgh fig helps to treat diabetes and high blood pressure',
                    'use 2':'(அத்திப்பழம்) ராக்ஸ்பர்க் அத்திப்பழம் நீரிழிவு மற்றும் உயர் இரத்த அழுத்தத்திற்கு சிகிச்சையளிக்க உதவுகிறது'
                    },
                'Ficus Religiosa (Peepal Tree)': {
                    'use 1':'Chewing the roots of a peepal tree is said to help prevent gum disease. The extract of leaves is used to heal wounds due to cuts or burns. The root bark extract has anti-ulcer and blood sugar lowering properties. Its leaf and bark extract are used to relieve toothaches and reduce swelling. So it cures Parkinson’s disease and bronchial diseases.',
                    'use 2':'(அரச மரம்) அரச மரம் மரத்தின் வேர்களை மென்று சாப்பிடுவது ஈறு நோயைத் தடுக்க உதவும் என்று கூறப்படுகிறது. இலைகளின் சாறு வெட்டுக்கள் அல்லது தீக்காயங்களால் ஏற்படும் காயங்களை குணப்படுத்த பயன்படுகிறது. வேர்ப்பட்டை சாற்றில் அல்சர் எதிர்ப்பு மற்றும் இரத்த சர்க்கரையை குறைக்கும் தன்மை உள்ளது. இதன் இலை மற்றும் பட்டை சாறு பல்வலியைப் போக்கவும் வீக்கத்தைக் குறைக்கவும் பயன்படுகிறது. எனவே இது பார்கின்சன் நோய் மற்றும் மூச்சுக்குழாய் நோய்களை குணப்படுத்துகிறது.'
                    },
                'Hibiscus Rosa-sinensis	': {
                    'use 1':'Hibiscus is used for treating loss of appetite, colds, heart and nerve diseases, upper respiratory tract pain and swelling (inflammation), fluid retention, stomach irritation, and disorders of circulation; for dissolving phlegm; as a gentle laxative; and as a diuretic to increase urine output.',
                    'use 2':'(செம்பருத்தி) பசியின்மை, சளி, இதயம் மற்றும் நரம்பு நோய்கள், மேல் சுவாசக்குழாய் வலி மற்றும் வீக்கம் (அழற்சி), வயிற்றில் எரிச்சல் மற்றும் சுழற்சி கோளாறுகள் ஆகியவற்றிற்கு செம்பருத்தி பயன்படுத்தப்படுகிறது; சளியை கரைப்பதற்கு; ஒரு மென்மையான மலமிளக்கியாக; மேலும் சிறுநீர் வெளியேற்றத்தை அதிகரிக்க ஒரு டையூரிடிக்.'
                    },
                'Jasminum (Jasmine)':{
                    'use 1': 'Jasmine is used on the skin to reduce the amount of breast milk, for skin diseases, and to speed up wound healing. Jasmine is inhaled to improve to  reduce stress, and reduce food cravings. In foods, jasmine is used to flavor beverages, frozen dairy desserts, candy, baked goods, gelatins, and puddings.',
                    'use 2':'(மல்லிகை) தாய்ப்பாலின் அளவைக் குறைக்கவும், தோல் நோய்களுக்கு, காயம் விரைவாக குணமடையவும் மல்லிகை தோலில் பயன்படுத்தப்படுகிறது. மன அழுத்தத்தைக் குறைக்கவும், உணவுப் பசியைக் குறைக்கவும் ஜாஸ்மின் உள்ளிழுக்கப்படுகிறது. உணவுகளில், மல்லிகை பானங்கள், உறைந்த பால் இனிப்புகள், மிட்டாய்கள், வேகவைத்த பொருட்கள், ஜெலட்டின்கள் மற்றும் புட்டிங்ஸ் ஆகியவற்றை சுவைக்க பயன்படுத்தப்படுகிறது'
                     },           
                'Mangifera Indica (Mango)':{
                    'use 1':'Mango helps to improved immunity and digestive and eye health',
                    'use 2':'(மாம்பழம்) மாம்பழம் நோய் எதிர்ப்பு சக்தி மற்றும் செரிமானம் மற்றும் கண் ஆரோக்கியத்தை மேம்படுத்த உதவுகிறது'
                    },
                'Mentha (Mint)':{
                    'use 1':'Mint reduce irritable bower syndrome, against upset stomachs, inhibit bacterial growth, treat fevers, flatulence, spastic colon',
                    'use 2':'(புதினா) புதினா எரிச்சலூட்டும் போவர் நோய்க்குறியைக் குறைக்கிறது, வயிற்று வலிக்கு எதிராக, பாக்டீரியா வளர்ச்சியைத் தடுக்கிறது, காய்ச்சல், வாய்வு மற்றும் ஸ்பாஸ்டிக் பெருங்குடலுக்கு சிகிச்சையளிக்கிறது'
                    },
                'Moringa Oleifera (Drumstick)':{
                    'use 1':'Drumstick control diabetes and high blood pressure fortifies the bone, improves skin health, treats erectile dysfunction and enhances libido. ',
                    'use 2':'(முருங்கை) முருங்கை இலை நீரிழிவு மற்றும் உயர் இரத்த அழுத்தம் எலும்பை பலப்படுத்துகிறது, தோல் ஆரோக்கியத்தை மேம்படுத்துகிறது, விறைப்புத்தன்மைக்கு சிகிச்சையளிக்கிறது'
                    },           
                'Muntingia Calabura (Jamaica Cherry-Gasagase)':{
                    'use 1' :' headaches, prostate problems, reduce gastric ulcers), bark (antiseptic), flowers (antiseptic, reduce swelling, antispasmodic, and fruits (respiratory problems, antidiarrheic).',
                    'use 2':'(தேன் பழம்) தலைவலி, ப்ரோஸ்டேட் பிரச்சனைகள், இரைப்பை புண்களைக் குறைக்கும், பட்டை (ஆன்டிசெப்டிக்), பூக்கள் (ஆன்டிசெப்டிக், வீக்கத்தைக் குறைக்கும், ஸ்பாஸ்மோடிக், மற்றும் பழங்கள் (சுவாச பிரச்சனைகள், வயிற்றுப்போக்கு)'
                    },
                'Murraya Koenigii (Curry)': {
                    'use 1':'Curry helps in the treatment of dysentery, diarrhea, diabetes, morning sickness, and nausea',
                    'use 2':'(கறிவேப்பிலை) வயிற்றுப்போக்கு, நீரிழிவு நோய், காலை நோய் மற்றும் குமட்டல் ஆகியவற்றின் சிகிச்சையில் கறிவேப்பிலை உதவுகிறது.'
                    }, 
                'Nerium Oleander (Oleander)':{
                    'use 1':'conditions, asthma, epilepsy, cancer, painful menstrual periods, leprosy, malaria, ringworm, indigestion, and venereal disease; and to cause abortions.',
                    'use 2':'(அரளிப்பூ) ஆஸ்துமா, கால்-கை வலிப்பு, புற்றுநோய், வலிமிகுந்த மாதவிடாய் காலம், தொழுநோய், மலேரியா, ரிங்வோர்ம், அஜீரணம் மற்றும் கருக்கலைப்புகளை ஏற்படுத்தவும் அரளிப்பூ பயன்படுத்தப்படுகிறது.'
                    },
                'Nyctanthes Arbor-tristis (Parijata)':{
                    'use 1': 'Parijat leaves has been used to treat a different kind of fevers, cough, arthritis, worm infestation',
                    'use 2':'(பவளமல்லி) பாரிஜாத இலைகள் பல்வேறு வகையான காய்ச்சல், இருமல், மூட்டுவலி, புழு தொல்லைக்கு சிகிச்சையளிக்கப் பயன்படுத்தப்படுகின்றன.'
                    },
                'Ocimum Tenuiflorum (Tulsi)':{
                    'use 1':'Tulsi used for Indigestion, Heart health, Respiratory Diseases, heart disease and fever',
                    'use 2':'(துளசி) துளசி அஜீரணம், இதய ஆரோக்கியம், சுவாச நோய்கள், இதய நோய் மற்றும் காய்ச்சலுக்கு பயன்படுத்தப்படுகிறது'
                    },
                'Piper Betle (Betel)':{
                    'use 1': 'Betal helps to Obesity, Hyperlipidaemia, Diabetes, Irregular Menstruation',
                    'use 2':'(வெற்றிலை) உடல் பருமன், ஹைப்பர்லிபிடேமியா, நீரிழிவு நோய், ஒழுங்கற்ற மாதவிடாய்க்கு வெற்றிலை உதவுகிறது.'
                    },
                'Plectranthus Amboinicus (Mexican Mint)':{
                    'use 1': 'skin, cleanse the body, protect against cough and cold, lessen joint discomfort, lessen stress, and promote digestion.',
                    'use 2':'(கற்பூரவல்லி) தோல், உடலைச் சுத்தப்படுத்துதல், இருமல் மற்றும் சளிக்கு எதிராகப் பாதுகாத்தல், மூட்டு அசௌகரியத்தைக் குறைத்தல், மன அழுத்தத்தைக் குறைத்தல் மற்றும் செரிமானத்தை மேம்படுத்துதல்.'
                    },
                'Pongamia Pinnata (Indian Beech)':{
                    'use 1': 'Beech used for tumors, piles, skin diseases, and ulcers',
                    'use 2':'(புங்கை) கட்டிகள், தோல் நோய்கள் மற்றும் புண்களுக்கு புங்கை பயன்படுத்தப்படுகிறது'
                    },
                'Psidium Guajava (Guava)':{
                    'use 1': 'Guava used for diarrhea, dysentery, stomach aches, and indigestion',
                    'use 2':'(கொய்யா) கொய்யாப்பழம் வயிற்றுப்போக்கு, வயிற்றுப்போக்கு, வயிற்று வலி மற்றும் அஜீரணத்திற்கு பயன்படுத்தப்படும்'
                    },
                'Punica Granatum (Pomegranate)':{
                    'use 1':'Pomegranate treats ulcers, diarrhea, and male infertility.',
                    'use 2':'(மாதுளை பழம்)மாதுளை புண்கள், வயிற்றுப்போக்கு மற்றும் ஆண் மலட்டுத்தன்மையை குணப்படுத்துகிறது.'
                    },
                'Santalum Album (Sandalwood)':{
                    'use 1': 'Sandalwood treats cold, urinary tract infections, digestive ',
                    'use 2':'(சந்தனம்) சந்தனம் சளி, சிறுநீர் பாதை நோய்த்தொற்றுகள், செரிமானத்தை குணப்படுத்துகிறது'
                    },
                'Syzygium Cumini (Jamun)':{
                    'use 1': 'Jamun helps to throats, asthma, bronchitis, thirst, biliousness, ulcers, and dysentery',
                    'use 2':'(நாவல்) தொண்டை, ஆஸ்துமா, மூச்சுக்குழாய் அழற்சி, தாகம், பித்தம், புண்கள் மற்றும் வயிற்றுப்போக்கு ஆகியவற்றிற்கு நாவல் உதவுகிறது.'
                    },
                'Syzygium jambos (Rose apple)': {
                    'use 1':'control diabetes, reduce toxicity and boosts immune system',
                    'use 2':'(ரோஸ் ஆப்பிள்) நீரிழிவு நோயைக் கட்டுப்படுத்துகிறது, நச்சுத்தன்மையைக் குறைக்கிறது மற்றும் நோய் எதிர்ப்பு சக்தியை அதிகரிக்கிறது'
                    },
                'Tabernaemontana Divaricata (Crape Jasmine)':{
                    'use 1': 'snake and scorpion poisoning',
                    'use 2':'(நந்தியாவட்டை) பாம்பு மற்றும் தேள் விஷத்தை நீக்க பயன்படுகிறது'
                    },
                'Phyllanthus emblica(Amala)':{
                    'use 1':'Amala used for Hair Care,Reduces Stress, Eye Care, Respiratory Health, Treats Anemia, Blood Purifier, Diuretic',
                    'use 2':'(நெல்லிமரம்)நெல்லி முடி பராமரிப்புக்கு பயன்படுத்தப்படுகிறது, மன அழுத்தத்தை குறைக்கிறது, கண் பராமரிப்பு, சுவாச ஆரோக்கியம், இரத்த சோகைக்கு சிகிச்சையளிக்கிறது.'
                    },
                'Tinospora cordifolia(Amruthaballi)':{
                    'use 1':'Amruthaballi treats for diabetes, urinary tract conditions, kidney infections, asthma, cardiac conditions',
                    'use 2':'(அமிர்தவல்லி) அம்ருதபல்லி நீரிழிவு நோய், சிறுநீர் பாதை நிலைமைகள், சிறுநீரக நோய்த்தொற்றுகள், ஆஸ்துமா, இதய நோய்களுக்கு சிகிச்சையளிக்கிறது'
                    },
                'Eclipta prostrata( Bhringraj)':{
                    'use 1':'Bhringraj effective used as a liver cleanser and act as a hair tonic',
                    'use 2':'(கரிசலாங்கண்ணி) கல்லீரலை சுத்தப்படுத்தியாகவும், முடி டானிக்காகவும் பயன்படுகிறது'
                    },
                'Coleus amboinicus(Doddapatre)':{
                    'use 1':'Doddapatre improve the health of your skin, detoxify the body, defend against colds, ease the pain of arthritis, relieve stress and anxiety, treat certain kinds of cancer, and optimize digestion.',
                    'use 2':'(ஓமவல்லி/கற்பூரவல்லி) நமது சருமத்தின் ஆரோக்கியத்தை மேம்படுத்துகிறது, உடலை நச்சு நீக்குகிறது, சளிக்கு எதிராக பாதுகாக்கிறது, மூட்டுவலியின் வலியை குறைக்கிறது, மன அழுத்தம் மற்றும் பதட்டத்தை போக்குகிறது, சில வகையான புற்றுநோய்களுக்கு சிகிச்சையளிக்கிறது மற்றும் செரிமானத்தை மேம்படுத்துகிறது'
                    },
                'Eucalyptus teriticornis(Eucalyptus)':{
                    'use 1':'Eucalyptus Treats a sore throat, sinusitis, and bronchitis',
                    'use 2':'(யூகலிப்டஸ்) யூகலிப்டஸ் தொண்டை புண், சைனசிடிஸ் மற்றும் மூச்சுக்குழாய் அழற்சிக்கு சிகிச்சையளிக்கிறது'
                    },
                'costus igneus(Insulin)':{
                    'use 1':'treatment and management of diabetes mellitus type-1 and sometimes diabetes mellitus type-2',
                    'use 2':'(இன்சுலின்) இன்சுலின் சிகிச்சை மற்றும் மேலாண்மை நீரிழிவு நோய் வகை-1 மற்றும் சில நேரங்களில் நீரிழிவு நோய் வகை-2'
                    },
                'Andrographis paniculata(Nilavembu)':{
                    'use 1':'Nilavembu helps to manage blood sugar levels and is useful for people suffering from diabetes.treats cancer and detoxifies the liver. Its rich source of antimicrobial and antiviral properties help manage all kinds of fever including dengue, typhoid, influenza, malaria and chikungunya.',
                    'use 2':'(நிலவேம்பு) நிலவேம்பு இரத்த சர்க்கரை அளவை நிர்வகிக்க உதவுகிறது மற்றும் நீரிழிவு நோயால் பாதிக்கப்பட்டவர்களுக்கு பயனுள்ளதாக இருக்கும். புற்றுநோய்க்கு சிகிச்சையளிக்கிறது மற்றும் கல்லீரலை நச்சு நீக்குகிறது. டெங்கு, டைபாய்டு, காய்ச்சல், மலேரியா மற்றும் சிக்குன்குனியா உள்ளிட்ட அனைத்து வகையான காய்ச்சலுக்கும் சிகிச்சை அளிக்க உதவுகிறது. '
                    },
                'Curcuma longa(Turmeric)':{
                    'use 1':'Turmeric used for disorders of the skin, upper respiratory tract, joints, and digestive system',
                    'use 2':'(மஞ்சள்) மஞ்சள் தோல், மேல் சுவாசக்குழாய், மூட்டுகள் மற்றும் செரிமான அமைப்பு ஆகியவற்றின் கோளாறுகளுக்குப் பயன்படுத்தப்படுகிறது.'
                    },
                'Cymbopogon(Lemongrass)':{
                    'use 1':'Lemongrass treating digestive tract spasms, stomachache, high blood pressure, convulsions, pain, vomiting, cough, achy joints (rheumatism), fever, the common cold, and exhaustion. It is also used to kill germs and as a mild astringent.',
                    'use 2':'(எலுமிச்சை புல்) ஜீரண மண்டல பிடிப்பு, வயிற்றுவலி, உயர் இரத்த அழுத்தம், வலிப்பு, வலி, வாந்தி, இருமல், மூட்டுவலி (வாத நோய்), காய்ச்சல், ஜலதோஷம் மற்றும் சோர்வு போன்றவற்றுக்கு எலுமிச்சைப் புல் சிகிச்சை அளிக்கிறது. இது கிருமிகளைக் கொல்லவும், லேசான துவர்ப்பு மருந்தாகவும் பயன்படுகிறது.'
                    },
                'Trigonella Foenum-graecum (Fenugreek)':{
                    'use 1':'Fenugreek treats carminative, gastric stimulant, antidiabetic and galactogogue',
                    'use 2':'(வெந்தயம்) வெந்தயம் கார்மினேடிவ், இரைப்பை தூண்டுதல், நீரிழிவு எதிர்ப்பு மற்றும் கேலக்டோகோக் ஆகியவற்றைக் கையாளுகிறது' 
                    }
        # Add more class names and their corresponding uses or information as needed
        }
    # Check if the class_name exists in the uses_dict
     #if class_name in uses_dict:
        #return uses_dict[class_name]
     #else:
      #  return "Information not available for this class."

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    #ef translate_to_tamil(text):
     #   translated_text = translator.translate(text, dest='ta').text
      #  return translated_text


    contnt = "<p>A medicinal plant is that species of the plant kingdom " \
            "Plants parts like flowers, leaves, roots, stems, fruits, or seeds are directly used or used in some preparation as a medicine to treat a condition or disease." \
            "<p>Applications of image processing and computer vision " \
            "techniques for the identification of the medicinal plants are very crucial as many of them are under " \
             "extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial " \
             "Information about the identified medicinal plants, including their names, Scientific name, and traditional uses</p>"
    image_url="https://media.cnn.com/api/v1/images/stellar/prod/230223174854-best-plant-identification-apps-lead-cnnu.jpg?c=original"
    st.image(image_url, use_column_width=True)
    if __name__ == '__main__':
    #add_bg_from_local("artifacts/Background.jpg")
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Identification of Medicinal plant</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown(contnt, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            img = Image.open(uploaded_file)

            img = img.resize((300, 300))
            st.image(img)

            if st.button("Predict"):
                pred = predict(img)
                name = class_dict[pred]
                if name in uses_dict:
                    uses_info = uses_dict[name]
                    result = f'<p style="font-family:sans-serif; color:Black; font-size: 20px;">The given image is {name}. '
                    for use_key, use_description in uses_info.items():
                        result += f'{use_description}<br>'
                        result += '\n \n'
                    result += '</p>'
                else:
        # If the plant name is not found in the uses_dict, provide a default message
                    result = f'<p style="font-family:sans-serif; color:Black; font-size: 16px;">The given image is {name}. Information not available for this plant.</p>'

                st.markdown(result, unsafe_allow_html=True)
                





if selected == "Remedies":
    #st.title(f"You have selected {selected}")
    st.markdown(
        '<h1 style="color: green;"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpg6wbC1dGFZaQwT8VoFq4FnU6UqmsbPD-IQW3pKZOAU5hd-LfUQM8EStFoye0UIhW8os&usqp=CAU" style="width: 100px;">𝓐𝔂𝓾𝓻𝓢𝓬𝓪𝓷 </h1>', 
        unsafe_allow_html=True
    )
    #st.markdown('<h1 style="color: green;">Identification of Medicinal plant using Machine learning</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: green;">Here are some common home remedies for various treatment based on plants</h3>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: green;">Home Remedies</h3>', unsafe_allow_html=True)
    st.write("Many home remedies exist that may help treat a variety of things, such as colds, inflammation, and pain. These are not always supported by research. But, scientists suggest that some may indeed work.")
    st.write("Chances are you’ve used a home remedy at some point: herbal teas for a cold, essential oils to dull a headache, plant-based supplements for a better night’s sleep. Maybe it was your grandma or you read about it online. The point is you tried it — and perhaps now you’re thinking, “Should I try it again?”") 
    image_url = "https://clinqonindia.com/wp-content/uploads/2023/05/Home-remedy-for-skin-fungal-infection.png"
    st.image(image_url, use_column_width=True)
    st.write("Here are some common home remedies for various ailments:")
    def home_page():
        medicinal_plants = [
            {
                "name": "Betel for digestion",
                "image_url": "https://thumbs.dreamstime.com/b/ripe-areca-nuts-betel-leaves-tray-illustration-nut-chewed-bowl-red-lime-golden-asian-traditional-chewing-39742995.jpg",
                "description": "chew betel leaves after meals; this can help with digestion. It may stimulate the secretion of digestive juices, relieve constipation and aid in reducing stomach bloating."
            },
            {
                "name": "Hibiscus Oil For Hair Nourishment",
                "image_url": "https://media.istockphoto.com/id/1389628600/vector/hibiscus-red-flower-with-leaves-closeup-isolated-on-white-background-vector-illustration.jpg?s=612x612&w=0&k=20&c=pqaK9syQojsF2sYPVnB5ay86dDaEu06PoVPwmOS83uo=",
                "description": "To prepare Hibiscus oil at home, take about 8 hibiscus flowers and 8 hibiscus leaves and grind them into a fine paste. Heat about a cup of coconut oil and add the paste to it. Let the mixture heat together and then keep it aside to cool down.Your hibiscus oil is ready to use. Massage your scalp with it for about 10 minutes and leave it on for about 30 minutes for best results.After that, wash your hair and scalp with a mild cleanser."
            },
            {
                "name": "Lemon tea",
                "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJM6euX2g8DArWxZA681wuXGypWX0JMorQWA&usqp=CAU",
                "description": "Tear the lemon leaves to release the flavour. Add all the ingredients to a pot. Bring the water to a boil and then pour it into the pot. Let the ingredients steep in the pot for 10-15 minutes. Pour the tea and drink. You can also add honey to sweeten it. "
            },
            {
                "name": "Guava tea for Diabetes",
                "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRiVN5emL5xD412ZRYutXusBm2-WAWxeKZmBA&usqp=CAU",
                "description": "Heat the water up until it boils, then turn off the stove. Add the leaves and allow to soak for 3 to 5 minutes. Then strain and take a sitz bath with this tea, carefully washing all the genital area. Repeat this procedure 2 to 3 times a day.."
            },
            {
                "name": "Fenugreek for dysentery",
                "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLjs8vMc4oz8oMiG9Lc2eH8LufnZPB85mjr7ZLvNPLe1PJC71Mjs93MwqZH_IbxkSgIns&usqp=CAU",
                "description": "Place some of these leaves in water and leave it overnight; next morning, strain the water and consume it."
            },
            {
                "name": "Tulsi kashayam ",
                "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRj03BPBfmXBtp_nzHHpmW2NaqEsEp357eVwEgNOfqeCZcsFY3vBC1w2K38lowwslrSSeY&usqp=CAU",
                "description": "Beat the cumin seeds and pepper in a mortar, Next, heat the ghee and the spices in a pan on medium flame, Add the toor dal water and bring it to a boil. Wait for the consistency, Put the green tulsi leaves and immediately turn off the stove, serve as hot "
            },
            {
                "name": "Parijat for Diabetes",
                "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8buZ_piHrDhyKqveg5y7P8m2jFCRutBaJJqWunkVzSam-mR2pGc4daBf_mo5qefo4GNs&usqp=CAU",
                "description": "Take Parijat leaves powder 1-3 gm or as directed by the physician Mix it with lukewarm water and have it once or twice a day."
            },
            {
                "name": "Neem to cure Chickenpox",
                "image_url": "https://t4.ftcdn.net/jpg/03/45/40/89/360_F_345408951_smPvYhyN291hSYt9ONVb1zrLctvBHvYz.jpg",
                "description": "Prepare a paste of neem leaves and apply it on your skin. Leave it on the skin for as long as you can. If your skin begins to itch, you can apply the neem paste again and leave it to dry. But it is recommended that you apply this paste only after taking a bath with neem leaves water. "
            },
            {
                "name": "Nilavembu Kashayam for fever",
                "image_url": "data:image/jpeg;base64,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",
                "description": "Boil 3 tablespoons (10 grams) of Nilavembu Kashayam herbal powder in 240 ml of water. Strain the concoction. Consume 60ml of it twice every day on an empty stomach Add jaggery or honey to boost taste."
            },
            {
                "name": "Eucalyptus Oil",
                "image_url": "data:image/jpeg;base64,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",
                "description": "Smelling and breathing in essential oils are the most common ways to enjoy aromatherapy. You can add eucalyptus essential oil to a diffuser or vaporizer or breathe in eucalyptus steam, which might help with coughing and congestion."
            }
        ]

        for plant in medicinal_plants:
            st.subheader(plant["name"])
            st.image(plant["image_url"], caption=plant["description"], width=500)
    # Display medicinal plants in two columns
       

# Main function to run the Streamlit app
    def main():
    # Add background image
    #add_bg_from_local("Background.jpg")
        home_page()

# Run the main function
    if __name__ == "__main__":
        main()





