import streamlit as st, sys, os, time
st.title("DEBUG APP (port 8503)")
st.write("python:", sys.executable)
st.write("cwd:", os.getcwd())
st.write("time:", time.ctime())
st.write("listing files in folder:")
for f in sorted(os.listdir(".")):
    st.write("-", f)
st.write("done")
