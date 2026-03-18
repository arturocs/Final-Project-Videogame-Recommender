

def submain():
    codigo="""
    def saludar():
        print ("Saludos invocador")
        """
    st.code(codigo, language="python")
    st.tittle("Se intenta")
    opciones=st.selectbox('Elige tu droga favorita',['Lol', 'Counter', 'Steam' ])
    st.write('Tu droga favortiva es:', {opciones})