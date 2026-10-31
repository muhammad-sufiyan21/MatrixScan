MATRIXSCAN PROJECT
==================

Files:
- MatrixScan_Linear_Algebra_Presentation.pptx : 11-slide presentation
- scanner_app.py : complete Python Streamlit web app
- requirements.txt : Python packages

Run locally:
1. Install Python 3.
2. Open Terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: streamlit run scanner_app.py
5. Open the local URL shown by Streamlit.

Important:
- The app asks the user to enter the four document-corner coordinates manually.
- It performs perspective correction, offers grayscale/black-and-white enhancement,
  and provides PDF and PNG downloads.
- The PPTX has a final source-code slide with a relative link to scanner_app.py.
  Keep the PPTX and scanner_app.py in the same folder for the local file link.
- To let classmates open the code online, upload scanner_app.py to a public GitHub
  repository and replace the link on the final slide with the repository URL.
- To publish the web app, deploy scanner_app.py on Streamlit Community Cloud.
