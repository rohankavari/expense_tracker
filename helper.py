import pandas as pd
import xml.etree.ElementTree as ET
from infer import get_sms_type
import re

def clean_text(text):
    # Define regex pattern to match non-English characters and symbols
    pattern = r'[^a-zA-Z0-9\s]'
    # Replace non-English characters and symbols with an empty string
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text

def xml_to_csv(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)

    # Get the root element of the XML tree
    root = tree.getroot()
    addresses = []
    bodies = []

    for sms in root.findall('sms'):
        address = sms.get('address')
        addresses.append(address)

        body = sms.get('body')
        bodies.append(clean_text(body.replace("\n","").replace(",","")))
        
    df = pd.DataFrame({'address': addresses, 'body': bodies})
    df.to_csv('dataset/mum.csv')

def int_to_class(x):
    lable={
        0:"spam",
        1:"otp",
        2:"banking",
        3:"bill"
    }
    return lable[x]

def class_to_int(cls):
    lable={
        "spam":0,
        "otp":1,
        "banking":2,
        "bill":3
    }
    return lable[cls]

def csv_predict(input_csv):
    df=pd.read_csv(input_csv)
    # Apply inference function to each row and store results in a new column
    df['class_id'] = df.apply(lambda row: get_sms_type( row['body'],row['address']), axis=1)

    # Convert class IDs to class names and store in another new column
    df['class_name'] = df['class_id'].apply(int_to_class)
    df.to_csv("dataset/mum_infer.csv")

if __name__=="__main__":
    # xml_to_csv('dataset/sms-20240401104853.xml')
    csv_predict("dataset/mum.csv")
