import streamlit as st
import subprocess
#Login Credentials
# Security

def main():
    menu=["Home","Login"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Home":
        st.subheader("Diabetes Analysis, Early Symptoms and Prediction")
        st.subheader("Individual Project by Jana Madi")
        from PIL import Image
        img = Image.open('Diabetes2.png')
        st.image(img,width=None)


    elif choice=="Login":
        st.subheader("Login Section")


        username=st.text_input('User Name')
        password=st.text_input('Password',type='password')

        if st.button("Login"):
                st.success ("Logged in as {}".format(username))
                subprocess.Popen(["streamlit", "run", "mental_health.py"])

    else:
            st.warning("Incorrect Username/Password")


if __name__ == '__main__':
    main()
