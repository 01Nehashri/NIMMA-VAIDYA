import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array,array_to_img
from keras.applications.mobilenet import MobileNet,preprocess_input 
from streamlit_chat import message
from joblib import load
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from keras import backend as K
import pywhatkit
from datetime import datetime
from textblob import TextBlob
def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )
df=pd.read_csv('District_Wise_Mental_Health_Patients_2021-22.csv')
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

            html, body, [class*="css"]  {
            font-family: 'Georgia', sans-serif;
            }
            </style>
            """

st.markdown(streamlit_style, unsafe_allow_html=True)


st.title("👩🏻‍⚕️ NIMMA VAIDYA")



menu = ["🏠 Home","👨🏻‍⚕️ Doctor","🙍 Patient"]

choice = st.sidebar.selectbox("MENU",menu)
if choice == "🏠 Home":
    st.subheader("THE NEW AGE DOCTOR")
    st.markdown("Rather than searching the Internet, what about using a symptom-checking app you may have loaded on your Laptop? Welcome to the one-stop website for all your medical needs! There are several inbuilt features at your disposal such as BMI Calculator, Mental Health Analysis, Skin Disease Check and several useful analytics which bring the concept of the 'New Age Doctor' to life.")
    st.image('Modern-doctor-500x333.jpg',width=700)
    
if choice == "👨🏻‍⚕️ Doctor":
    st.subheader("HELLO DOCTOR!")
    menu1 = ["Brain Tumor Prediction","Informative Plots"]
    ch = st.selectbox("Select a Domain",menu1)
    st.text("")
    if ch == "Brain Tumor Prediction":
        model = tf.keras.models.load_model("final_model.h5")
        ### load file
        uploaded_file = st.file_uploader("Choose an image file", type="jpg")
    
        map_dict = {0: 'Glioma',
                1: 'Meningioma',
                2: 'No Tumor',
                3: 'Pituitary'}

        if uploaded_file is not None:
        # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(opencv_image,(224,224))
            # Now do something with the image! For example, let's display it:
            st.image(opencv_image, channels="RGB")

            resized = preprocess_input(resized)
            img_reshape = resized[np.newaxis,...]

            Generate_pred = st.button("Generate Prediction")    
            if Generate_pred:
                prediction = model.predict(img_reshape).argmax()
                st.title("Predicted label for the image is {}".format(map_dict [prediction]))
                st.text("")

    if ch==("Informative Plots"):
        fig1=px.bar(df,x='DISTRICT ',y='Total',title='Total Patients by District')
        st.plotly_chart(fig1)
        fig=px.pie(df,values='SUICIDE_ATTEMPT_CASES',names='DISTRICT ',title='Suicide attempts by district')
        st.plotly_chart(fig) 
        

if choice=="🙍 Patient":
    st.subheader("HELLO USER!")
    menu2 = ["Skin Disease Prediction","Mental Health Prediction","BMI Prediction","Informative Plots","FAQs"]
    ch2 = st.selectbox("Select a Domain:",menu2)
    st.text("")
    st.text("")
    st.text("")
    if ch2 == "Skin Disease Prediction":
        model1 = tf.keras.models.load_model("final_model_skin (1).h5")
        uploaded_file1 = st.file_uploader("Upload an Image to detect the skin disease", type="jpg")

        map_dict1 = {0: 'possibility of benign skin disease',
                 1: 'possiblity of malignant skin disease'}
        



        if uploaded_file1 is not None:
        # Convert the file to an opencv image.
            file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
            opencv_image1 = cv2.imdecode(file_bytes1, 1)
            opencv_image1 = cv2.cvtColor(opencv_image1, cv2.COLOR_BGR2RGB)
            resized1 = cv2.resize(opencv_image1,(224,224))
            # Now do something with the image! For example, let's display it:
            st.image(opencv_image1, channels="RGB")

            resized1 = preprocess_input(resized1)
            img_reshape1 = resized1[np.newaxis,...]
            st.text("")
            Genrate_pred1 = st.button("Generate Result")
            prediction1=0
            if Genrate_pred1:
                prediction1 = model1.predict(img_reshape1).argmax()
                st.title("Predicted Label for the image is {}".format(map_dict1[prediction1]))
        
            menu4 = ["Area1","Area2","Area3"]
            ch1 = st.selectbox("Select area to check for doctors nearby",menu4)
            if ch1=="Area1":
                st.write("CONTACT DETAILS: Doctor1")
            elif ch1=="Area2":
                st.write("CONTACT DETAILS: Doctor2")
            else:
                st.write("CONTACT DETAILS: Doctor3")


    if ch2 == "Mental Health Prediction":
        from_sent=st.text_input("How are you feeling?")
        if st.button('Predict'):
            br=TextBlob(from_sent)
            result = br.sentiment.polarity
            if result==0.15:
                st.success("You are doing well!")
            elif result>0.15:
                st.success("You are doing very well!")
            else:
                st.write("Here are 5 things you can do to feel better. They may seem simple, but they can help a lot.")
                st.write("")
                st.write("1. Exercise")
                st.write("2. Eat healthy foods and drink plenty of water")
                st.write("3. Express yourself")
                st.write("4. Don't dwell on problems")
                st.write("5. Notice good things")

    if ch2 == "BMI Prediction":
        
        input_filename = st.file_uploader("Upload an Image to check the BMI", type="jpg")  
        
        input_shape = (224, 224, 3)
        base_model = keras.applications.ResNet152(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg"
        )

        base_model.trainable = False
        
        if input_filename is not None:
        # Convert the file to an opencv image.
        
            
            file_bytes = np.asarray(bytearray(input_filename.read()), dtype=np.uint8)
            raw_input_image = cv2.imdecode(file_bytes, 1)
            raw_input_image = cv2.cvtColor(raw_input_image, cv2.COLOR_BGR2RGB)
            raw_input_image = cv2.resize(raw_input_image, (input_shape[0], input_shape[1]))
            # Now do something with the image! For example, let's display it:
            st.image(raw_input_image, channels="RGB")
            
            plt.figure(0)
            plt.imshow(raw_input_image)

            preprocessed_input_image = load_img(input_filename, target_size=input_shape)
            preprocessed_input_image = img_to_array(preprocessed_input_image)
            plt.figure(1)
            plt.imshow(preprocessed_input_image.astype(int))

            plt.figure(2)
            preprocessed_input_image[preprocessed_input_image[:,:,0] > 0] = 1
            preprocessed_input_image[preprocessed_input_image[:,:,1] > 0] = 1
            preprocessed_input_image[preprocessed_input_image[:,:,2] > 0] = 1
            plt.imshow(preprocessed_input_image)

            plt.figure(3)
            final_input_image = raw_input_image * preprocessed_input_image
            plt.imshow(final_input_image.astype(int))


            test_datagen = image.ImageDataGenerator(samplewise_center=True,)

            generator = test_datagen.flow(np.expand_dims(final_input_image, axis=0),batch_size=1)
            features_batch = base_model.predict(generator)

            dependencies = {
                'coeff_determination': coeff_determination
            }
            model2 = keras.models.load_model('3.935_model.h5', custom_objects=dependencies)
            preds = model2.predict(features_batch)
            bmi_pred = preds[0][0]
            
            st.write(f"BMI: {bmi_pred}")
            if bmi_pred < 16:
              st.write("Very severely underweight")
            elif 16 <= bmi_pred < 17:
              st.write("Severely underweight")
            elif 17 <= bmi_pred < 19:
              st.write("Underweight")
            elif 19<= bmi_pred < 27:
              st.write("Normal")
            elif 27 <= bmi_pred < 31:
              st.write("Overweight")
            elif 31 <= bmi_pred < 36:
              st.write("Moderately obese")
            elif 36<= bmi_pred < 41:
              st.write("Severely obese")
            elif bmi_pred >= 41:
              st.write("Very severely obese")

    if ch2 == "Informative Plots":
        df1=pd.read_csv("food.csv")
        fig3=px.bar(df1,x='Food_items',y=['Proteins','Carbohydrates'],title='Nutrition Chart')
        st.plotly_chart(fig3)
        fig4=px.scatter(df1, x='Fats', y='Calories',title='Calories v Fats')
        st.plotly_chart(fig4)
    if ch2 == "FAQs":
        if st.button('How do the disease-causing germs invade my body?'):
            st.markdown('Your skin is a wonderful protective barrier that prevents many of the disease-causing germs that you run into each day from entering your body. Only when you have an opening in your skin—like a cut or a scrape—are germs likely to enter there. Most germs enter through your mouth and nose, making their way farther into your body through your respiratory or digestive tracts. ')
            
        if st.button('What is the difference between bacteria and viruses?'):
            st.markdown('Bacteria are single-celled organisms that have the ability to feed themselves and to reproduce. They are found everywhere, including the air, water, and soil. They divide and multiply very quickly, which means that one cell can become 1 million cells in just a few hours. Viruses are microorganisms that are smaller than bacteria, but they cannot grow or reproduce without the help of a separate living cell. Once a virus gets inside your body, it attaches itself to a healthy cell and uses the cell’s nucleus to reproduce itself.')
            
        if st.button('Why is exercise important to health?'):
            st.markdown('Exercise is good for your health. Regular physical activity helps a person have stronger bones and muscles, helps control body fat, helps prevent certain illnesses, and contributes to a good outlook on life. Regular exercise helps promote digestion and a good night’s sleep. When children exercise as part of their busy lives, they are better equipped to manage the physical and emotional challenges of a busy day.')
            
            
      

        