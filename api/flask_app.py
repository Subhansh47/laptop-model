from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('pipemodel.pkl', 'rb'))
df = pd.read_csv('laptopPrice.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    brands = sorted(df['brand'].unique())
    processor_brands = sorted(df['processor_brand'].unique())
    processor_names = sorted(df['processor_name'].unique())
    processor_gnrtns = sorted(df['processor_gnrtn'].unique())
    ram_gbs = sorted(df['ram_gb'].unique())
    ram_types = sorted(df['ram_type'].unique())
    ssds = sorted(df['ssd'].unique())
    hdds = sorted(df['hdd'].unique())
    oss = sorted(df['os'].unique())
    os_bits = sorted(df['os_bit'].unique())
    graphic_card_gbs = sorted(df['graphic_card_gb'].unique())
    weights = sorted(df['weight'].unique())
    warranties = sorted(df['warranty'].unique())
    touchscreens = sorted(df['Touchscreen'].unique())
    msoffices = sorted(df['msoffice'].unique())
    
    return render_template('index.html', brands=brands, processor_brands=processor_brands,
                           processor_names=processor_names, processor_gnrtns=processor_gnrtns,
                           ram_gbs=ram_gbs, ram_types=ram_types, ssds=ssds, hdds=hdds, oss=oss,
                           os_bits=os_bits, graphic_card_gbs=graphic_card_gbs, weights=weights,
                           warranties=warranties, touchscreens=touchscreens, msoffices=msoffices)

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form.get('brand')
    processor_brand = request.form.get('processor_brand')
    processor_name = request.form.get('processor_name')
    processor_gnrtn = request.form.get('processor_gnrtn')
    ram_gb = request.form.get('ram_gb')
    ram_type = request.form.get('ram_type')
    ssd = request.form.get('ssd')
    hdd = request.form.get('hdd')
    os = request.form.get('os')
    os_bit = request.form.get('os_bit')
    graphic_card_gb = request.form.get('graphic_card_gb')
    weight = request.form.get('weight')
    warranty = request.form.get('warranty')
    touchscreen = request.form.get('touchscreen')
    msoffice = request.form.get('msoffice')

    x=[[brand,processor_brand,processor_name,processor_gnrtn,ram_gb,ram_type,ssd,hdd,os,os_bit,graphic_card_gb,weight,warranty,touchscreen,msoffice]]
    print(x)

    new_data = pd.DataFrame(x,
                        columns=['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
                                 'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit', 'graphic_card_gb',
                                 'weight', 'warranty', 'Touchscreen', 'msoffice'])
    # x=pd.DataFrame(input_data)
    prediction = model.predict(new_data)[0]
    return str(np.round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
