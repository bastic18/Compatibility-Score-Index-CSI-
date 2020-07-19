# Compatibility-Score-Index

Starter Code for Capstone Project

Steps to properly install dependencies on local machine are as follows in VS Code Terminal:


## DATABASE SETUP:

N.B the name of the database is called "csi_capstone", ensure there is no other database on your system with that name
The database used is MariaDb (Flask mysqldb is used for communication with the database)
To set up the database, the CSI.sql needs to be execute from the mariaDb command line interface:

+ Use the command \\.{path of file}\CSI.sql  to run the CSI.sql file
+ {path of file} refers to the full path location of the file on the computer.
+ For example,  \\. C:\Users\basti\Documents\Compatibility-Score-Index\app\CSI.sql


## VIRTUAL ENVIRONMENT CREATION & ACTIVATION:
1.  Check python version:  
    python --version

    Python 3 and over should be installed on your local machine

2.  #### Create Virtual Environment :
+   *cd Compatibility-Score-Index*
+   *python3 -m venv venv*

We need to change directory from the root directory to the Scripts folder inside the venv directory
The commands to do this are listed below

1.   **cd venv**
2.   **cd Scripts**
3.   **./activate** 
4.   Use the **cd..** command to go back to root folder
###### OR WE CAN DO FROM THE ROOT FOLDER:
1. **venv/Scripts/Activate**
+   (venv) should appear before Folder path in the terminal



## INSTALL REQUIREMENTS.TXT:
1. from the root folder use the command **pip install -r requirements.txt**


## BACKEND SETUP:

1. Upon sucessful creation of virtual environment, activate the environment with:
+    *venv/Scripts/Activate*

2. Proceed to install these specified packages using the commands below:
+   *pip install tensorflow==1.14*
+   *pip install spacy*
+   *python -m spacy download en_core_web_lg*
+   *pip install joblib*
+   *pip install keras==2.2.4*
+   *pip install h5py* 
+   *pip install sklearn* 

3. Finally, execute the project with:
+   *python run.py*

4. Visit localhost:8080 in browser

### FILE PATHS
1. Ensure that file paths match to your local machine. For example in CSI_code.py line 904 
+ loaded_model_age = joblib.load(r'C:\Users\basti\Documents\Compatibility-Score-Index\app\finalized_model_age.sav')
+ should be:  
+ loaded_model_age = joblib.load(r'{local machine directory}\Compatibility-Score-Index\app\finalized_model_age.sav')

### The changes are expected to be in init.py and CSI_code.py
